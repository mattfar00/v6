# ---------------------------------------------------------------------------
# STORIA LUNGA — livello dati comune per Fondo e PAC (Blocco B)
# ---------------------------------------------------------------------------
# Scarica e normalizza le serie storiche lunghe usate per caratterizzare gli
# ETF dell'anagrafica (data/anagrafica_etf.json) e, in prospettiva, i
# benchmark dei comparti del fondo. PRINCIPIO: solo esposizioni STATICHE
# (asset class larghe: azionario per area, obbligazionario aggregate/gov,
# oro). Gli ETF settoriali/tematici sono esclusi by design: la loro
# esposizione cambia nel tempo e non e' ricostruibile con un proxy fisso.
#
# Fonti (tutte gratuite, scaricate a runtime con cache; un CSV in
# data/storia_lunga/<serie>.csv ha SEMPRE la precedenza — formato: due
# colonne "date,value" con value = LIVELLO/prezzo, non rendimento):
#   - Shiller (mirror GitHub 'datasets/s-and-p-500'): S&P composite,
#     dividendi, CPI USA, tassi 10Y — mensile dal 1871.
#   - Yahoo Finance (yfinance): fondi comuni USA a storia lunga con NAV
#     total-return (VFINX 1976, VEURX 1990, VEIEX 1994, VBMFX 1986,
#     QQQ 1999, VUSXX 1992...).
#   - FRED (endpoint CSV): oro LBMA dal 1968, CPI Italia dal ~1955.
# Ogni funzione ritorna pd.Series MENSILE di RENDIMENTI SEMPLICI, indice
# fine-mese, oppure None se la fonte non risponde (mai eccezioni in UI).
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st

CARTELLA_LOCALE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "storia_lunga")

URL_SHILLER = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
URL_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# Ticker Yahoo dei proxy a storia lunga (NAV aggiustato = total return).
PROXY_YAHOO = {
    "vfinx": "VFINX",   # S&P 500, dal 1976 (mutual fund: Yahoo a volte non lo serve)
    "veurx": "VEURX",   # Europa sviluppata (USD), dal 1990
    "veiex": "VEIEX",   # Mercati emergenti (USD), dal 1994
    "vbmfx": "VBMFX",   # Aggregate bond USA (USD), dal 1986
    "rpibx": "RPIBX",   # Bond internazionali NON coperti, dal 1986
    "qqq":   "QQQ",     # Nasdaq-100 TR, dal 1999
    "vusxx": "VUSXX",   # T-bill/cash, dal 1992
    "spy":   "SPY",     # S&P 500 ETF TR, dal 1993 — fallback azionario affidabile
    "agg":   "AGG",     # Aggregate bond ETF, dal 2003 — fallback obbligazionario
}
# Serie FRED: (id primario, eventuali fallback)
PROXY_FRED = {
    "gold_lbma": ("GOLDAMGBD228NLBM", ["GOLDPMGBD228NLBM"]),
    "cpi_it": ("ITACPIALLMINMEI", ["CPALTT01ITM661N"]),
    # Cambio spot USD per 1 EUR (DEXUSEU), trattato come una serie di
    # "prezzo" qualunque: carica_fred() lo converte in rendimenti mensili
    # (variazione % di quanti USD vale un EUR) con lo stesso meccanismo delle
    # altre serie FRED. Serve per tradurre in EUR le serie proxy quotate in
    # USD (es. gold_lbma) — diverso dall'hedging: qui si TRADUCE il prezzo
    # con il cambio spot realizzato, non si rimuove il rischio cambio.
    "eur_usd_fx": ("DEXUSEU", []),
}

# Tassi brevi (LIVELLI annualizzati, es. 0.045 = 4.5%, NON serie di rendimento)
# usati solo per la copertura cambio sintetica USD/EUR (vedi
# costruisci_serie_hedged_eur). Id primario + fallback.
PROXY_FRED_TASSI = {
    "us_3m": ("TB3MS", ["DTB3"]),                       # T-bill USA 3 mesi
    "eur_3m": ("IR3TIB01EZM156N", ["EURIBOR3MD_N.B"]),  # Interbancario 3 mesi area euro
}

# Pesi del composite azionario globale (esposizione STATICA dichiarata,
# ~MSCI ACWI odierno senza Pacifico). Rinormalizzati sulle serie disponibili
# in ciascun mese (pre-1994 niente EM, pre-1990 solo USA).
PESI_WORLD_COMPOSITE = {"vfinx": 0.60, "veurx": 0.30, "veiex": 0.10}


# ---------------------------------------------------------------------------
# UTILITÀ
# ---------------------------------------------------------------------------
def _da_livelli_a_rendimenti(livelli: pd.Series) -> pd.Series:
    livelli = livelli.dropna().astype(float)
    livelli = livelli.resample("ME").last().dropna()
    return livelli.pct_change().dropna()


def _csv_locale(nome: str):
    """Override manuale: data/storia_lunga/<nome>.csv con colonne date,value."""
    percorso = os.path.join(CARTELLA_LOCALE, f"{nome}.csv")
    if not os.path.isfile(percorso):
        return None
    df = pd.read_csv(percorso)
    df.columns = [c.strip().lower() for c in df.columns]
    serie = pd.Series(df["value"].values,
                      index=pd.to_datetime(df["date"]), name=nome)
    return _da_livelli_a_rendimenti(serie)


# Ultimo errore/diagnostica di download per serie, per spiegare i fallimenti
# in UI invece di ingoiarli in silenzio (nome -> messaggio).
ULTIMO_ERRORE_YAHOO = {}


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def carica_yahoo_max(nome: str):
    """Storico massimo mensile total-return di un proxy Yahoo. None se fallisce.

    Prova in ordine: CSV locale -> download diretto mensile -> Ticker.history
    mensile -> download giornaliero ricampionato a mensile noi stessi. Per
    VFINX in particolare i dati esistono su Yahoo (verificato a mano: storico
    visibile dal 1985) ma il bucket mensile dell'API chart a volte non
    risponde: il tentativo 3 (giornaliero, poi ricampionato qui) copre
    proprio questo caso, prima di arrendersi e lasciare che il chiamante usi
    un fallback (es. SPY).
    """
    loc = _csv_locale(nome)
    if loc is not None:
        return loc
    ticker = PROXY_YAHOO[nome]
    ULTIMO_ERRORE_YAHOO.pop(nome, None)
    try:
        import yfinance as yf
    except Exception as e:
        ULTIMO_ERRORE_YAHOO[nome] = f"yfinance non disponibile: {e}"
        return None

    def _da_dati(data):
        if data is None or data.empty:
            return None
        col = data["Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        out = _da_livelli_a_rendimenti(col)
        return out if len(out) else None

    try:
        data = yf.download(ticker, period="max", interval="1mo",
                           progress=False, auto_adjust=True, actions=False)
        out = _da_dati(data)
        if out is not None:
            return out
    except Exception as e:
        ULTIMO_ERRORE_YAHOO[nome] = f"download 1mo: {e}"

    try:
        data = yf.Ticker(ticker).history(period="max", interval="1mo",
                                         auto_adjust=True, actions=False)
        out = _da_dati(data)
        if out is not None:
            return out
    except Exception as e:
        ULTIMO_ERRORE_YAHOO[nome] = f"Ticker.history 1mo: {e}"

    try:
        data = yf.download(ticker, period="max", interval="1d",
                           progress=False, auto_adjust=True, actions=False)
        out = _da_dati(data)
        if out is not None:
            return out
        ULTIMO_ERRORE_YAHOO[nome] = "nessun dato nemmeno a livello giornaliero"
    except Exception as e:
        ULTIMO_ERRORE_YAHOO[nome] = f"download 1d: {e}"

    return None


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def carica_fred(nome: str):
    """Serie FRED (livelli) -> rendimenti/variazioni mensili. None se fallisce."""
    loc = _csv_locale(nome)
    if loc is not None:
        return loc
    sid, fallback = PROXY_FRED[nome]
    for s in [sid] + list(fallback):
        try:
            df = pd.read_csv(URL_FRED.format(sid=s))
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            serie = pd.Series(df["value"].values,
                              index=pd.to_datetime(df["date"]), name=nome)
            out = _da_livelli_a_rendimenti(serie)
            if len(out) > 24:
                return out
        except Exception:
            continue
    # Fallback per l'ORO: le serie LBMA su FRED sono state dismesse (2024).
    # Yahoo: futures COMEX GC=F dal 2000, poi ETC GLD dal 2004.
    if nome == "gold_lbma":
        try:
            import yfinance as yf
            for tk in ("GC=F", "GLD"):
                data = yf.download(tk, period="max", interval="1mo",
                                   progress=False, auto_adjust=True,
                                   actions=False)
                if data is None or data.empty:
                    continue
                col = data["Close"]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                out = _da_livelli_a_rendimenti(col)
                if len(out) > 24:
                    return out
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def _carica_fred_livello(nome_tasso: str):
    """
    Serie FRED di TASSO (livello annualizzato, es. 0.045 = 4.5%), NON
    convertita in rendimenti — a differenza di carica_fred(), qui il dato è
    già il numero che serve (un tasso d'interesse, non un prezzo). Ritorna
    pd.Series mensile (frazione, non %) o None se la fonte non risponde.
    Un CSV in data/storia_lunga/<nome_tasso>.csv (date,value con value in %,
    es. 4.5) ha la precedenza, come per le altre serie.
    """
    percorso = os.path.join(CARTELLA_LOCALE, f"{nome_tasso}.csv")
    if os.path.isfile(percorso):
        df = pd.read_csv(percorso)
        df.columns = [c.strip().lower() for c in df.columns]
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                      index=pd.to_datetime(df["date"]), name=nome_tasso) / 100.0
        return s.resample("ME").last().dropna()
    sid, fallback = PROXY_FRED_TASSI[nome_tasso]
    for s_id in [sid] + list(fallback):
        try:
            df = pd.read_csv(URL_FRED.format(sid=s_id))
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            serie = pd.Series(df["value"].values,
                              index=pd.to_datetime(df["date"]), name=nome_tasso) / 100.0
            out = serie.resample("ME").last().dropna()
            if len(out) > 24:
                return out
        except Exception:
            continue
    return None


def costruisci_serie_hedged_eur(nome_base: str, base: pd.Series):
    """
    Approssima il rendimento mensile di un proxy USD non-coperto COME SE
    fosse coperto in cambio verso l'euro (currency-hedged share class),
    usando la parità coperta dei tassi d'interesse — SENZA bisogno del
    tasso di cambio spot, solo dei tassi brevi USD/EUR:

        hedged = (1 + rendimento_usd) * (1 + tasso_eur/12) / (1 + tasso_usd/12) - 1

    Derivazione: un hedge mensile "roll" a termine blocca il cambio al
    forward F invece dello spot futuro; per la parità coperta dei tassi
    F/S ≈ (1+i_usd/12)/(1+i_eur/12), quindi il valore in EUR coperto al
    mese successivo è il valore USD moltiplicato per (1+i_eur/12)/(1+i_usd/12)
    — lo spot esce dai conti, resta solo il differenziale tassi (che è
    esattamente il "costo/guadagno di copertura" già citato in
    anagrafica_etf.json per RPIBX/AGGH: storicamente 0-3%/anno).

    APPROSSIMAZIONE, non replica esatta: usa tassi 3 mesi come proxy del
    costo di un roll mensile, ignora il basis cross-currency (differenza
    residua tra costo di hedging realizzato e la parità teorica, che nella
    realtà è quasi sempre piccola ma non nulla). Utile per avvicinare la
    volatilità/rendimento di indici "€ hedged" (es. benchmark Cometa/Fon.Te)
    quando l'unico proxy disponibile è un fondo USD non coperto.

    Copre solo dal ~1999 (nascita euro / disponibilità EURIBOR): per mesi
    precedenti la serie risultante e' piu' corta dell'originale non-coperta.

    `base`: la serie di rendimenti USD non coperta già caricata (es.
    serie["world_composite"] o serie["vbmfx"] da carica_tutte_le_serie), così
    non serve ricaricarla né dipendere da uno stato globale condiviso.
    Ritorna pd.Series di rendimenti mensili, o None se i tassi non sono
    disponibili o l'overlap è insufficiente.
    """
    loc = _csv_locale(f"{nome_base}_eur_hedged")
    if loc is not None:
        return loc
    if base is None:
        return None
    r_usd = _carica_fred_livello("us_3m")
    r_eur = _carica_fred_livello("eur_3m")
    if r_usd is None or r_eur is None:
        return None
    df = pd.concat([base, r_usd.rename("r_usd"), r_eur.rename("r_eur")],
                   axis=1, join="inner").dropna()
    if len(df) < 24:
        return None
    fattore = (1 + df["r_eur"] / 12.0) / (1 + df["r_usd"] / 12.0)
    hedged = (1 + df.iloc[:, 0]) * fattore - 1
    hedged.name = f"{nome_base}_eur_hedged"
    return hedged


def costruisci_serie_eur_da_usd(nome_base: str, base: pd.Series, fx_return: pd.Series):
    """
    TRADUCE (non copre) in EUR una serie di rendimenti quotata in USD, usando
    il cambio spot REALIZZATO — l'opposto concettuale di
    costruisci_serie_hedged_eur: qui il rischio cambio non viene rimosso, si
    ricalcola solo il rendimento come lo vedrebbe un investitore che parte e
    arriva in EUR (nessuna copertura). Utile per confrontare correttamente
    proxy USD-quotati (es. gold_lbma, prezzo dell'oro in USD/oncia da FRED)
    con ETF quotati in EUR sullo stesso sottostante (es. SGLD.MI, oro fisico
    in EUR): senza questa conversione la "correlazione" misurata include
    anche il rumore EUR/USD, che non è vera divergenza sul sottostante.

    Derivazione: se P_usd è il prezzo in USD e S le quotazioni USD-per-EUR
    (es. FRED DEXUSEU), il prezzo in EUR è P_eur = P_usd / S, quindi
        r_eur = (1 + r_usd) / (1 + r_fx) - 1
    dove r_fx è la variazione mensile di S (r_fx>0 = EUR si è apprezzato:
    penalizza il rendimento EUR di un asset USD, come atteso).

    `fx_return`: la serie di rendimento mensile di "eur_usd_fx" (già la
    variazione % di DEXUSEU, calcolata da carica_fred come qualunque altra
    serie di livelli).

    Ritorna pd.Series di rendimenti mensili, o None se manca la base o il
    cambio, o l'overlap è insufficiente.
    """
    loc = _csv_locale(f"{nome_base}_eur")
    if loc is not None:
        return loc
    if base is None or fx_return is None:
        return None
    df = pd.concat([base, fx_return.rename("fx")], axis=1, join="inner").dropna()
    if len(df) < 24:
        return None
    tradotta = (1 + df.iloc[:, 0]) / (1 + df["fx"]) - 1
    tradotta.name = f"{nome_base}_eur"
    return tradotta


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def carica_shiller():
    """
    Dataset Shiller (mensile dal 1871). Ritorna dict di pd.Series:
      sp500_tr   rendimenti mensili S&P TOTAL RETURN (prezzo + dividendo/12)
      cpi_us     variazione mensile CPI USA
      bond10_tr  rendimenti mensili RICOSTRUITI del decennale USA:
                 r ~ y/12 + Dmod*(y_prec - y_curr), Dmod = duration modificata.
                 E' una ricostruzione (no dati di prezzo reali): usarla per
                 ANCORE e sanity check, non come verita' mensile fine.
    None se la fonte non risponde.
    """
    try:
        loc = os.path.join(CARTELLA_LOCALE, "shiller.csv")
        df = pd.read_csv(loc) if os.path.isfile(loc) else pd.read_csv(URL_SHILLER)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        p = df["SP500"].astype(float)
        d = df["Dividend"].astype(float)
        tr = (p + d / 12.0) / p.shift(1) - 1.0
        sp500_tr = tr.dropna()
        sp500_tr.index = sp500_tr.index + pd.offsets.MonthEnd(0)

        cpi = df["Consumer Price Index"].astype(float)
        cpi_us = cpi.pct_change().dropna()
        cpi_us.index = cpi_us.index + pd.offsets.MonthEnd(0)

        y = df["Long Interest Rate"].astype(float) / 100.0
        dmod = (1.0 - (1.0 + y) ** -10) / y.replace(0, np.nan)
        bond = (y.shift(1) / 12.0 + dmod * (y.shift(1) - y)).dropna()
        bond.index = bond.index + pd.offsets.MonthEnd(0)

        return {"sp500_tr": sp500_tr, "cpi_us": cpi_us, "bond10_tr": bond}
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def costruisci_world_composite():
    """
    Azionario globale a ESPOSIZIONE STATICA: 60% USA + 30% Europa + 10% EM,
    pesi rinormalizzati ogni mese sulle serie disponibili (pre-1994 senza EM,
    pre-1990 solo USA). Approssimazione dichiarata di MSCI World/ACWI; un
    data/storia_lunga/world_composite.csv (es. MSCI World ufficiale) la
    sostituisce integralmente.
    """
    loc = _csv_locale("world_composite")
    if loc is not None:
        return loc
    basi = {k: carica_yahoo_max(k) for k in PESI_WORLD_COMPOSITE}
    if basi.get("vfinx") is None:
        # FALLBACK: Yahoo spesso non serve più VFINX (mutual fund storico,
        # ticker discontinuato). Sostituiamo solo la componente USA con SPY
        # (stesso indice, TR dal 1993) invece di scartare l'intero composite:
        # così Europa/EM restano nel mix invece di perdere tutto il peso 40%.
        basi["vfinx"] = carica_yahoo_max("spy")
    basi = {k: v for k, v in basi.items() if v is not None}
    if not basi:
        return None
    df = pd.DataFrame(basi)
    pesi = pd.DataFrame(
        {k: np.where(df[k].notna(), PESI_WORLD_COMPOSITE[k], 0.0) for k in df},
        index=df.index)
    somma = pesi.sum(axis=1)
    out = (df.fillna(0.0) * pesi).sum(axis=1) / somma.replace(0, np.nan)
    return out.dropna()


# ---------------------------------------------------------------------------
# REGISTRO SERIE — punto d'accesso unico per i motori (Fondo e PAC)
# ---------------------------------------------------------------------------
def carica_tutte_le_serie():
    """
    Ritorna (serie, note): serie = dict nome -> pd.Series di rendimenti
    mensili (senza intersezione: ogni serie tiene la SUA lunghezza),
    note = dict nome -> stringa fonte/avviso. Include il CPI per i reali.
    """
    serie, note = {}, {}

    sh = carica_shiller()
    if sh is not None:
        serie["shiller_sp500"] = sh["sp500_tr"]
        serie["cpi_us"] = sh["cpi_us"]
        serie["bond10y_usa_ricostruito"] = sh["bond10_tr"]
        note["shiller_sp500"] = "Shiller/GitHub, TR dal 1871"
        note["cpi_us"] = "CPI USA (Shiller)"
        note["bond10y_usa_ricostruito"] = ("RICOSTRUZIONE dai tassi 10Y "
                                           "(solo ancore/sanity check)")
    else:
        note["shiller_sp500"] = "⚠️ fonte non raggiungibile"

    # spy/agg sono fallback interni (non proxy propri: nessuna riga dedicata,
    # entrano solo al posto di vfinx/vbmfx se Yahoo non li serve).
    FALLBACK_YAHOO = {"vfinx": ("spy", "SPY, S&P 500 TR dal 1993"),
                      "vbmfx": ("agg", "AGG, Aggregate bond TR dal 2003")}
    for nome in PROXY_YAHOO:
        if nome in ("spy", "agg"):
            continue
        s = carica_yahoo_max(nome)
        if s is not None:
            serie[nome] = s
            note[nome] = f"Yahoo {PROXY_YAHOO[nome]}, NAV total return"
        elif nome in FALLBACK_YAHOO:
            diag = ULTIMO_ERRORE_YAHOO.get(nome, "")
            alt_nome, alt_desc = FALLBACK_YAHOO[nome]
            s_alt = carica_yahoo_max(alt_nome)
            if s_alt is not None:
                serie[nome] = s_alt
                note[nome] = (f"⚠️ Yahoo {PROXY_YAHOO[nome]} non scaricato"
                              f"{f' ({diag})' if diag else ''} → "
                              f"fallback {alt_desc}")
            else:
                note[nome] = (f"⚠️ Yahoo {PROXY_YAHOO[nome]} non scaricato"
                              f"{f' ({diag})' if diag else ''}")
        else:
            diag = ULTIMO_ERRORE_YAHOO.get(nome, "")
            note[nome] = (f"⚠️ Yahoo {PROXY_YAHOO[nome]} non scaricato"
                          f"{f' ({diag})' if diag else ''}")

    for nome in PROXY_FRED:
        s = carica_fred(nome)
        if s is not None:
            serie[nome] = s
            note[nome] = f"FRED {PROXY_FRED[nome][0]}"
        else:
            note[nome] = (f"⚠️ FRED {PROXY_FRED[nome][0]} non disponibile "
                          f"(per l'oro il fallback Yahoo GC=F/GLD è automatico) — "
                          f"drop-in: data/storia_lunga/{nome}.csv")

    wc = costruisci_world_composite()
    if wc is not None:
        serie["world_composite"] = wc
        note["world_composite"] = ("composite statico 60/30/10 "
                                   "USA/Europa/EM (rinormalizzato)")

    # Versioni "€ hedged" sintetiche (parità coperta dei tassi USD/EUR): per
    # i benchmark di fondo negoziale che dichiarano esplicitamente indici
    # "Total Return € hedged" (es. Cometa Crescita, Fon.Te), usare queste al
    # posto delle serie USD non coperte riduce il rumore di cambio che non
    # esiste nel benchmark reale. Copertura solo dal ~1999 (tassi EUR/EURIBOR
    # disponibili da lì); None se i tassi non si scaricano (nessun effetto
    # sugli altri usi delle serie base, che restano invariate).
    for base_nome in ("world_composite", "vbmfx"):
        base = serie.get(base_nome)
        if base is None:
            continue
        hedged = costruisci_serie_hedged_eur(base_nome, base)
        chiave = f"{base_nome}_eur_hedged"
        if hedged is not None:
            serie[chiave] = hedged
            note[chiave] = (f"sintetica: {base_nome} corretta per il "
                            "differenziale tassi USD/EUR (parità coperta "
                            "dei tassi, no cambio spot) — approssima un "
                            "indice € hedged, copertura dal ~1999")
        else:
            note[chiave] = ("⚠️ non calcolabile: mancano i tassi brevi "
                            "USD/EUR (FRED us_3m/eur_3m) o overlap "
                            "insufficiente")

    # Traduzione in EUR (cambio spot, non hedging) delle serie proxy quotate
    # in USD che vengono confrontate con ETF quotati in EUR sullo stesso
    # sottostante — oggi solo l'oro (gold_lbma è USD/oncia da FRED, mentre
    # ETC come SGLD.MI quotano in EUR): senza tradurlo, il beta/corr stimati
    # da estendi_classi_pac includono anche il rumore EUR/USD, non solo la
    # vera dinamica dell'oro. Fallback silenzioso sulla serie USD originale
    # se il cambio non è disponibile, così il consumatore (CLASSE_TO_SERIE)
    # trova sempre la chiave "gold_lbma_eur" quando c'è gold_lbma.
    gold_usd = serie.get("gold_lbma")
    if gold_usd is not None:
        fx = serie.get("eur_usd_fx")
        tradotto = costruisci_serie_eur_da_usd("gold_lbma", gold_usd, fx)
        if tradotto is not None:
            serie["gold_lbma_eur"] = tradotto
            note["gold_lbma_eur"] = ("sintetica: gold_lbma (USD/oncia, FRED) "
                                     "tradotto in EUR col cambio spot "
                                     "EUR/USD (FRED DEXUSEU, dal 1999)")
        else:
            serie["gold_lbma_eur"] = gold_usd
            note["gold_lbma_eur"] = ("⚠️ cambio EUR/USD non disponibile: "
                                     "fallback sulla serie USD originale "
                                     "(gold_lbma), non tradotta")
    return serie, note


def serie_reale(rend: pd.Series, cpi: pd.Series) -> pd.Series:
    """Deflaziona rendimenti mensili con l'inflazione mensile allineata."""
    df = pd.concat([rend, cpi], axis=1, join="inner").dropna()
    return (1 + df.iloc[:, 0]) / (1 + df.iloc[:, 1]) - 1


def _cagr(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return np.nan
    return float(np.exp(np.log1p(s).sum() * 12 / len(s)) - 1)


# ---------------------------------------------------------------------------
# TAB DI VERIFICA DATI (Blocco B: nessun motore toccato)
# ---------------------------------------------------------------------------
def render_tab_dati():
    st.subheader("📚 Storia lunga — livello dati comune (verifica)")
    st.caption(
        "Serie proxy a **esposizione statica** per caratterizzare gli ETF "
        "dell'anagrafica e i benchmark dei comparti. Nessuna intersezione: "
        "ogni serie conserva la propria lunghezza. Gli ETF settoriali/"
        "tematici sono esclusi by design (esposizione non ricostruibile). "
        "Un CSV in `data/storia_lunga/<serie>.csv` (date,value in livelli) "
        "sostituisce la fonte remota. Questi dati NON alimentano ancora i "
        "motori: prima si verificano qui (Blocco B), poi mapping e beta "
        "(Blocco C), poi i motori (Blocco D)."
    )
    with st.spinner("Scarico/leggo le serie lunghe (cache 7-30 giorni)..."):
        serie, note = carica_tutte_le_serie()

    if not serie:
        st.error("Nessuna serie caricata: ambiente senza rete? Usa i drop-in "
                 "CSV in data/storia_lunga/.")
        return

    cpi_us = serie.get("cpi_us")
    righe = []
    for nome, s in serie.items():
        if nome == "cpi_us":
            continue
        reale = serie_reale(s, cpi_us) if cpi_us is not None else None
        ann = (1 + s) .rolling(12).apply(np.prod, raw=True) - 1
        righe.append({
            "Serie": nome,
            "Fonte": note.get(nome, ""),
            "Da": s.index.min().strftime("%Y-%m"),
            "A": s.index.max().strftime("%Y-%m"),
            "Mesi": len(s),
            "CAGR nom. (%)": round(_cagr(s) * 100, 2),
            "CAGR reale USD (%)": (round(_cagr(reale) * 100, 2)
                                    if reale is not None and len(reale) else np.nan),
            "Vol (%)": round(float(s.std(ddof=1)) * np.sqrt(12) * 100, 1),
            "Peggior mese (%)": round(float(s.min()) * 100, 1),
            "Peggior 12m (%)": (round(float(ann.min()) * 100, 1)
                                 if len(ann.dropna()) else np.nan),
        })
    st.dataframe(pd.DataFrame(righe), use_container_width=True, hide_index=True)

    problemi = [f"{k}: {v}" for k, v in note.items() if v.startswith("⚠️")]
    if problemi:
        st.warning("Fonti non disponibili — " + " · ".join(problemi))

    st.caption(
        "Sanity check consigliati: CAGR reale azionario USA ~6-7% dal 1871; "
        "oro reale ~1-2% dal 1968 con vol ~15-20%; bond10y ricostruito reale "
        "~1,5-2,5%. Se vedi numeri lontani da questi, la fonte o il "
        "drop-in ha un problema. Il CAGR 'reale' qui usa il CPI USA (serie "
        "in USD); la deflazione in euro reali italiani (CPI Italia) entra "
        "nei motori al Blocco D."
    )
