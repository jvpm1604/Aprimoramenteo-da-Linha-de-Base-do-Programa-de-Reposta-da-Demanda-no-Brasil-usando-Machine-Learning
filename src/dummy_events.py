"""
Eventos fictícios (dummy events) — o motor da trilha de acurácia.

Ideia (CAISO/CalTRACK): avaliar os estimadores em janelas de dias SEM despacho
real. Ali a redução verdadeira é conhecidamente NULA, o que dá VERDADE DE CAMPO
— algo indisponível em eventos reais, onde o contrafactual não é observável.

Consequências:
  - o erro do estimador é diretamente mensurável (NMBE, CV(RMSE));
  - todo volume apurado por max(0, B - y) é crédito espúrio integral (erro puro).

Regras do desenho:
  - janela vespertina 14h-18h, com a janela de ajuste nas horas anteriores;
  - apenas dias úteis e sábados (a grade regulatória não cobre domingo/feriado);
  - apenas dias SEM evento em qualquer hora relevante;
  - apenas no conjunto de TESTE cronológico (fora da amostra de treino);
  - todos os estimadores são comparados nas MESMAS linhas (comparabilidade
    estrita — sem isso, um estimador poderia "ganhar" só por avaliar em horas
    mais fáceis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import inday, models
from .features import split_cronologico


def avaliar_agente(d: pd.DataFrame, h0: int = None, h1: int = None,
                   com_ml: bool = True) -> dict | None:
    """Roda os 6 estimadores para UM agente sobre os eventos fictícios.

    Retorna dict com as previsões alinhadas, o real e os metadados, ou None se
    o agente não tiver dados suficientes.
    """
    h0 = cfg.DUMMY_H0 if h0 is None else h0
    h1 = cfg.DUMMY_H1 if h1 is None else h1

    d = d.sort_values("DATETIME")
    d = d[d["CONS_MW"].notna()]
    if len(d) < 1500:
        return None

    tr, te = split_cronologico(d)
    if len(tr) < 500 or len(te) < 100:
        return None

    win = inday.janela_ajuste(h0)
    horas_ev = list(range(h0, h1 + 1))
    necessarias = win + horas_ev

    R = te.pivot_table(index="data", columns="HORA", values="CONS_MW", aggfunc="mean")
    L = te.pivot_table(index="data", columns="HORA", values="Baseline_CCEE_fiel",
                       aggfunc="mean")
    E = te.pivot_table(index="data", columns="HORA",
                       values="SINALIZADOR_HORARIO_ATEND_PROD", aggfunc="max")

    if any(h not in R.columns or h not in L.columns for h in necessarias):
        return None

    ok = (
        L[horas_ev].notna().all(axis=1)
        & R[necessarias].notna().all(axis=1)
        & (E.reindex(columns=necessarias).fillna(0).sum(axis=1) == 0)  # sem evento
    )
    dias = R.index[ok]
    if len(dias) < 10:
        return None

    add, mul = inday.aplicar(R.loc[dias], L.loc[dias], h0)

    tev = te[te["data"].isin(dias) & te["HORA"].between(h0, h1)].copy()
    tev = tev.merge(add, left_on="data", right_index=True)
    tev = tev.merge(mul, left_on="data", right_index=True)

    P = {
        "1_estatica": tev["Baseline_CCEE_fiel"].values,
        "2_aditivo": (tev["Baseline_CCEE_fiel"] + tev["add"]).values,
        "3_multiplicativo": (tev["Baseline_CCEE_fiel"] * tev["mul"]).values,
    }

    if com_ml:
        m_ex = models.treinar_rf(tr, cfg.FEAT_EXANTE)
        m_full = models.treinar_rf(tr, cfg.FEAT_COMPLETO)
        towt = models.TOWT().fit(tr)
        if m_ex is not None:
            P["4_RF_exante"] = m_ex.predict(tev[cfg.FEAT_EXANTE].fillna(
                tr[cfg.FEAT_EXANTE].median()))
        if m_full is not None:
            P["5_RF_completo"] = m_full.predict(tev[cfg.FEAT_COMPLETO].fillna(
                tr[cfg.FEAT_COMPLETO].median()))
        P["6_TOWT"] = towt.predict(tev)

    # máscara COMUM: todos os estimadores avaliados nas mesmas observações
    y = tev["CONS_MW"].values
    comum = np.isfinite(y) & (y > 0)
    for v in P.values():
        comum &= np.isfinite(v)

    return {
        "y": y[comum],
        "pred": {k: v[comum] for k, v in P.items()},
        "n_dias": int(len(dias)),
        "n_obs": int(comum.sum()),
    }
