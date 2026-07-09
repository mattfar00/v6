# ---------------------------------------------------------------------------
# MODELLO LUNGO — estensione delle classi PAC via proxy storico lungo
# ---------------------------------------------------------------------------
# Usa il livello dati di storia_lunga.py per allungare all'indietro le classi
# del PAC (Azionario/Obbligazionario/Oro) quando lo storico reale dei ticker
# scelti è più corto del periodo simulato, tramite beta+alfa sull'overlap con
# un proxy a storia lunga (dal 1976-86). Le classi senza proxy affidabile
# (Immobiliare, Azioni singole) restano invariate.
#
# NOTA (lug-2026): questo modulo conteneva anche una ricostruzione analoga
# per i COMPARTI DEI FONDI PENSIONE dal benchmark dichiarato nel DPI (style
# analysis RBSA, alfa sull'overlap), più un intero tab "Modello & Validazione"
# (validazione out-of-sample, stress test deterministici, parametri
# suggeriti nu/kappa/shrinkage). È stato tutto rimosso su richiesta: per
# fondi a gestione ATTIVA (mandati con limite di tracking-error, non a
# replica) il benchmark dichiarato nel DPI non è garanzia di cosa il gestore
# ha davvero tenuto in portafoglio nel tempo — quindi estendere la storia di
# un comparto da quell'ipotesi (anche con pesi presi dal DPI ufficiale)
# resta un'assunzione non verificabile. Nessuna validazione a valle (R²,
# copertura P10-P90) rende affidabile un'ipotesi a monte che non lo è.
#
# Il PAC invece resta qui: lavora su ETF/ticker REALI a replica passiva
# (non fondi a gestione attiva con benchmark dichiarato), quindi il caso
# beta+alfa su overlap è più difendibile — decisione esplicita, non un
# automatismo silenzioso.
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

from storia_lunga import carica_tutte_le_serie

# Mapping classe PAC -> serie lunga (per l'estensione delle classi)
CLASSE_TO_SERIE = {
    "Azionario": "world_composite",
    "Obbligazionario": "vbmfx",
    # gold_lbma è quotato in USD/oncia (FRED); gli ETC come SGLD.MI quotano
    # in EUR. "gold_lbma_eur" è la stessa serie tradotta in EUR col cambio
    # spot EUR/USD (storia_lunga.costruisci_serie_eur_da_usd), altrimenti la
    # corr misurata includerebbe anche il rumore di cambio, non solo la
    # dinamica dell'oro. Fallback automatico su gold_lbma (USD) se il cambio
    # non è disponibile.
    "Oro/Materie prime": "gold_lbma_eur",
    # Immobiliare e Azioni singole: nessun proxy statico affidabile -> no ext.
}


def _serie_lunghe():
    serie, note = carica_tutte_le_serie()
    return serie, note


def estendi_classi_pac(serie_classi: dict):
    """
    serie_classi: dict nome_classe -> pd.Series rendimenti mensili (ETF).
    Estende ogni classe col proxy lungo (beta+alfa su overlap). Le classi
    senza proxy restano invariate. Ritorna (dict esteso, diagnostica).
    """
    lunghe, _ = _serie_lunghe()
    out, diag = {}, {}
    for classe, s in serie_classi.items():
        proxy_nome = CLASSE_TO_SERIE.get(classe)
        proxy = lunghe.get(proxy_nome) if proxy_nome else None
        if proxy is None:
            out[classe] = s
            diag[classe] = "nessun proxy statico: solo storico ETF"
            continue
        ov = pd.concat([s, proxy], axis=1, join="inner").dropna()
        if len(ov) < 24:
            out[classe] = s
            diag[classe] = f"overlap col proxy troppo corto ({len(ov)} mesi)"
            continue
        beta = float(np.cov(ov.iloc[:, 0], ov.iloc[:, 1])[0, 1] /
                     max(np.var(ov.iloc[:, 1]), 1e-12))
        alpha = float(ov.iloc[:, 0].mean() - beta * ov.iloc[:, 1].mean())
        corr = float(np.corrcoef(ov.iloc[:, 0], ov.iloc[:, 1])[0, 1])
        pre = (alpha + beta * proxy[proxy.index < s.index.min()])
        out[classe] = pd.concat([pre, s]).sort_index()
        diag[classe] = (f"proxy {proxy_nome}: beta {beta:.2f}, corr {corr:.2f}, "
                        f"{len(out[classe])} mesi totali (reali: {len(s)})")
    return out, diag
