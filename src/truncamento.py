"""
Script 06 — Trilha B: resiliência a ondas de calor (Tabela 4.5 do TCC).

Isola os N dias mais quentes da série, REMOVE-OS DO TREINO (blackout) e simula um
despacho vespertino. Roda SEM defasagens — condição realista, já que num evento
efetivo a recência limpa não está disponível.

O achado que corrige uma expectativa comum:
  - com a temperatura do evento ACIMA do máximo de treino, o Random Forest
    SATURA e falha (não extrapola: nenhuma folha aprendeu carga além do que viu);
  - o TOWT, com resposta térmica linear por partes, EXTRAPOLA e protege o agente.

Ou seja, a resiliência térmica não vem da sofisticação do algoritmo, mas da sua
capacidade de extrapolação.

RIGOR: os três estimadores são avaliados nas MESMAS linhas. Domingos entram nos
"dias mais quentes" mas não têm linha de base (a grade não os cobre) — se não
alinharmos, a CCEE seria avaliada em n menor que o RF/TOWT.

Uso:  python scripts/06_onda_calor.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as cfg
from src.metrics import avaliar
from src.models import TOWT, treinar_rf

def trilha_b(df, agente, n_dias=5, h0=None, h1=None):
    h0 = cfg.DUMMY_H0 if h0 is None else h0
    h1 = cfg.DUMMY_H1 if h1 is None else h1

    d = df[df["SIGLA_PERFIL_AGENTE"] == agente].sort_values("DATETIME")
    d = d[d["CONS_MW"].notna()]

    quentes = d.groupby("data")["temp_c"].max().sort_values(ascending=False).head(n_dias).index
    ev = d[d["data"].isin(quentes) & d["HORA"].between(h0, h1)].copy()
    tr = d[~d["data"].isin(quentes) & d["treino_valido"]].copy()

    print(f"\n=== TRILHA B — {agente} ===")
    print("dias quentes:", [str(x.date()) for x in quentes])
    t_tr, t_ev = tr["temp_c"].max(), ev["temp_c"].max()
    print(f"treino_max={t_tr:.1f}C | evento_max={t_ev:.1f}C"
          + ("   <-- RF PRECISA EXTRAPOLAR" if t_ev > t_tr + 0.1 else ""))

    rf = treinar_rf(tr, cfg.FEAT_EXANTE)
    tw = TOWT().fit(tr)

    ev["p_rf"] = rf.predict(ev[cfg.FEAT_EXANTE].fillna(tr[cfg.FEAT_EXANTE].median()))
    ev["p_towt"] = tw.predict(ev)

    # MESMAS LINHAS para os três (drop dos dias sem linha de base, ex.: domingo)
    ev = ev.dropna(subset=["Baseline_CCEE_fiel", "p_rf", "p_towt", "CONS_MW"])
    print(f"n avaliado (linhas identicas): {len(ev)}")

    out = []
    for nome, col in [("Estatica (CCEE)", "Baseline_CCEE_fiel"),
                      ("Random Forest", "p_rf"), ("TOWT", "p_towt")]:
        m = avaliar(ev["CONS_MW"], ev[col])
        print(f"  {nome:16s}: NMBE={m['NMBE']:>6.2f}  CVRMSE={m['CVRMSE']:>6.2f}")
        out.append(dict(agente=agente, estimador=nome, **m))
    return out

def main():
    df = pd.read_csv(cfg.DS_BASELINE, low_memory=False)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"]); df["data"] = pd.to_datetime(df["data"])
    res = []
    for ag in ["BOULEVARD RIO", "TOP SHOPPING"]:
        res += trilha_b(df, ag)
    t = pd.DataFrame(res)
    t.to_csv(cfg.OUT_TABLES / "06_onda_calor.csv", index=False)
    print(f"\n-> {cfg.OUT_TABLES}/06_onda_calor.csv")

if __name__ == "__main__":
    main()
