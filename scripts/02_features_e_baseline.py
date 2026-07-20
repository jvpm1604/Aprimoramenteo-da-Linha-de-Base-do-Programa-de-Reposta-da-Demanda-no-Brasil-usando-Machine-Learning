"""
Script 02 — Engenharia de atributos + reconstrução da linha de base da CCEE.

Entrada : data/interim/dataset_TCC_final_TCC.csv     (saída do script 00)
Saída   : data/processed/dataset_TCC_baseline_fiel.csv

Este é o script que produz o COMPARADOR DE STATUS QUO do trabalho: a linha de
base regulatória reconstruída fielmente à norma (REN 1.040/2022 + NT-ONS 0049),
e não uma aproximação simplificada. Sem isso, a comparação seria ilegítima.

Uso:
    python scripts/02_features_e_baseline.py                # caso base (M-2)
    python scripts/02_features_e_baseline.py --lag 1        # sensibilidade M-1
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as cfg
from src.baseline import calcular_baseline_ccee
from src.features import preparar

def main(lag: int, saida: pathlib.Path):
    print("Carregando e derivando atributos...")
    df = preparar()
    print(f"  {len(df):,} linhas | {df['SIGLA_PERFIL_AGENTE'].nunique()} agentes")
    print(f"  consumo espurio (zeros) -> NaN: {int(df['cons_espurio'].sum()):,}")
    print(f"  horas em dia de evento (blackout): {int(df['dia_evento'].sum()):,}")

    print(f"\nReconstruindo a linha de base (defasagem M-{lag})...")
    df = calcular_baseline_ccee(df, lag_meses=lag)

    cob = df["Baseline_CCEE_fiel"].notna().mean()
    grade = df["is_util"] | df["is_sab"]
    cob_grade = df.loc[grade, "Baseline_CCEE_fiel"].notna().mean()
    print(f"  cobertura geral            : {cob:6.1%}")
    print(f"  cobertura na grade util/sab: {cob_grade:6.1%}")
    print("  (domingos/feriados e o 1o mes de cada agente ficam SEM linha de")
    print("   base por construcao — comportamento correto, nao falha.)")

    df.to_csv(saida, index=False)
    print(f"\n-> {saida}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, default=cfg.LAG_LB_MESES,
                    help="defasagem de contabilizacao em meses (default: 2)")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out) if a.out else cfg.DS_BASELINE
    main(a.lag, out)
