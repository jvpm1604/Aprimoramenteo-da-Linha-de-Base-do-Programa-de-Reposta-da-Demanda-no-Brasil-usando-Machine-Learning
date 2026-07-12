"""
Script 04 — Tabela-degrau: os 6 estimadores nos arquétipos (Tabelas 4.1 e 4.2 do TCC).

Compara, EXATAMENTE NAS MESMAS OBSERVAÇÕES (eventos fictícios, 14h-18h):

  degrau (i)   1_estatica        -> linha de base da CCEE (status quo)
  degrau (ii)  2_aditivo         -> + ajuste intradiário (NESO/ISO-NE)   [PROPOSTA]
               3_multiplicativo  -> + ajuste intradiário (AEMO, ±20%)    [PROPOSTA]
  degrau (iii) 4_RF_exante       -> RF sem defasagens (mesma info que a média)
               5_RF_completo     -> RF com defasagens  (TETO de acurácia)
               6_TOWT            -> resposta térmica linear por partes

O propósito dos três degraus é ISOLAR A ORIGEM DO GANHO:
  (ii) vs (i)  = efeito do ajuste intradiário
  (iii) vs (ii)= quanto ainda seria recuperável por um modelo complexo

Uso:  python scripts/04_tabela_degrau.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as cfg
from src.dummy_events import avaliar_agente
from src.metrics import avaliar

def main():
    df = pd.read_csv(cfg.DS_BASELINE, low_memory=False)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["data"] = pd.to_datetime(df["data"])

    linhas = []
    for ag in cfg.ARQUETIPOS:
        r = avaliar_agente(df[df["SIGLA_PERFIL_AGENTE"] == ag])
        if r is None:
            print(f"{ag}: dados insuficientes"); continue
        print(f"\n===== {ag} | {cfg.DUMMY_H0}-{cfg.DUMMY_H1}h | "
              f"{r['n_dias']} dias | {r['n_obs']} obs identicas =====")
        for k in sorted(r["pred"]):
            m = avaliar(r["y"], r["pred"][k])
            print(f"  {k:17s}: NMBE={m['NMBE']:>7.2f}  CVRMSE={m['CVRMSE']:>6.2f}  MAPE={m['MAPE']:>6.2f}")
            linhas.append(dict(agente=ag, estimador=k, **m))

    t = pd.DataFrame(linhas)
    t.to_csv(cfg.OUT_TABLES / "04_tabela_degrau.csv", index=False)

    for met in ["CVRMSE", "NMBE"]:
        piv = t.pivot(index="estimador", columns="agente", values=met)
        piv = piv.reindex(columns=[a for a in cfg.ARQUETIPOS if a in piv.columns])
        print(f"\n--- {met} (%) ---"); print(piv.to_string())
        piv.to_csv(cfg.OUT_TABLES / f"04_degrau_{met.lower()}.csv")
    print(f"\n-> {cfg.OUT_TABLES}/04_*.csv")

if __name__ == "__main__":
    main()
