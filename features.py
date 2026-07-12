"""
Script 01 — Auditoria de integridade da base.

Motivo: toda a modelagem depende do ALINHAMENTO TEMPORAL. Uma falha silenciosa
de defasagem (comum quando há lacunas e o lag é calculado por deslocamento de
POSIÇÃO em vez de TEMPO) enviesaria todos os modelos sem gerar erro aparente.

Verifica, por agente:
  1. contiguidade horária (diferenças consecutivas == 1h);
  2. unicidade temporal (sem horários duplicados);
  3. alinhamento das defasagens: lag_k(t) == y(t-k), k in {1, 24, 168}.

Uso:  python scripts/01_auditoria_base.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as cfg
from src.features import carregar_base

def main():
    df = carregar_base()
    df["data"] = df["DATA_DT"].dt.normalize()
    df["DATETIME"] = df["data"] + pd.to_timedelta(df["HORA"], unit="h")

    print(f"Linhas: {len(df):,} | Agentes: {df['SIGLA_PERFIL_AGENTE'].nunique()}")
    print(f"Período: {df['DATA_DT'].min().date()} -> {df['DATA_DT'].max().date()}\n")

    linhas = []
    for ag, d in df.groupby("SIGLA_PERFIL_AGENTE"):
        d = d.sort_values("DATETIME")
        dt = d["DATETIME"].diff().dropna()
        gaps = int((dt != pd.Timedelta(hours=1)).sum())
        dups = int(d["DATETIME"].duplicated().sum())

        # alinhamento das defasagens: fração de casos em que lag_k == y.shift(k)
        checks = {}
        for k, col in [(1, "lag_1h"), (24, "lag_24h"), (168, "lag_168h")]:
            esperado = d["CONS_MW"].shift(k)
            m = esperado.notna() & d[col].notna()
            checks[col] = float(np.isclose(d.loc[m, col], esperado[m], atol=1e-6).mean()) if m.any() else np.nan

        linhas.append(dict(agente=ag, n=len(d), gaps=gaps, duplicatas=dups,
                           **{f"match_{c}": round(v, 4) for c, v in checks.items()}))

    aud = pd.DataFrame(linhas).sort_values("agente")
    problemas = aud[(aud["gaps"] > 0) | (aud["duplicatas"] > 0)
                    | (aud[["match_lag_1h", "match_lag_24h", "match_lag_168h"]] < 0.999).any(axis=1)]

    print(aud.to_string(index=False))
    print("\n" + ("OK: series contiguas, sem duplicatas, defasagens alinhadas."
                  if problemas.empty else f"ATENCAO: {len(problemas)} agente(s) com problema."))

    saida = cfg.OUT_TABLES / "01_auditoria_integridade.csv"
    aud.to_csv(saida, index=False)
    print(f"\n-> {saida}")

if __name__ == "__main__":
    main()
