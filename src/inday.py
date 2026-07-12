"""
Script 03 — Termossensibilidade e seleção dos arquétipos (Tabela 3.2 do TCC).

Correlação de Pearson entre consumo (MW) e temperatura de bulbo seco (°C), por
agente. Revela um espectro nítido:
  r > +0,5  -> comercial (shopping): carga dominada por climatização
  r ~ 0     -> indústria de batelada: regida pelo cronograma de produção
  r < 0     -> processo eletroquímico contínuo (levemente maior no frio)

Cruzado com o sinalizador de despacho da CCEE, define os arquétipos analisados.

Uso:  python scripts/03_termossensibilidade.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as cfg

def main():
    df = pd.read_csv(cfg.DS_BASELINE, low_memory=False)

    linhas = []
    for ag, d in df.groupby("SIGLA_PERFIL_AGENTE"):
        m = d["CONS_MW"].notna() & d["temp_c"].notna()
        ev = d[d["SINALIZADOR_HORARIO_ATEND_PROD"] > 0]
        linhas.append(dict(
            agente=ag,
            r_temp=round(d.loc[m, "CONS_MW"].corr(d.loc[m, "temp_c"]), 3),
            r_lag168=round(d["CONS_MW"].corr(d["lag_168h"]), 3),
            carga_media_MW=round(d["CONS_MW"].mean(), 1),
            horas_evento=len(ev),
            dias_evento=ev["data"].nunique(),
        ))

    t = pd.DataFrame(linhas).sort_values("r_temp", ascending=False)
    print(t.to_string(index=False))
    t.to_csv(cfg.OUT_TABLES / "03_termossensibilidade.csv", index=False)

    print("\nNOTA — r_lag168 (correlacao com a mesma hora da semana anterior) mede o")
    print("RITMO SEMANAL da carga. Onde ele e alto, a memoria recente carrega o sinal")
    print("preditivo; onde e baixo (batelada), nenhum lag preve bem. E fisica da carga.")
    print(f"\n-> {cfg.OUT_TABLES}/03_termossensibilidade.csv")

if __name__ == "__main__":
    main()
