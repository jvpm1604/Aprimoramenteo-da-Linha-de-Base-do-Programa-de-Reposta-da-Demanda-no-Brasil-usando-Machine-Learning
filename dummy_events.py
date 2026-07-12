"""
Script 00 — Ingestão e consolidação da base horária (produz dataset_TCC_final_TCC.csv).

  medição CCEE (SMF)  +  clima INMET  +  calendário  ->  uma linha por agente-hora

ATENÇÃO — este script é uma RECONSTRUÇÃO DOCUMENTADA do pipeline original (que
rodava no Google Colab sobre o Drive). Os leitores dos arquivos brutos dependem
do layout exato dos seus arquivos da CCEE e do INMET; ajuste `ler_ccee` e
`ler_inmet` conforme necessário. O contrato de SAÍDA, esse sim, é fixo — é o que
os scripts 01+ consomem:

    SIGLA_PERFIL_AGENTE; DATA_DT; HORA; CONS_MW;
    temp_c; umid_rel; rad_global; precip_mm; vento_vel;
    CATEGORIA; CLASSE; PRODUTO; OFERTA; MONTANTE_PRELIMINAR_RD;
    SINALIZADOR_HORARIO_ATEND_PROD; dia_semana; feriado;
    DATA_STR; CODIGO_PERFIL_AGENTE; lag_1h; lag_24h; lag_168h

DECISÕES QUE IMPORTAM (e por quê):
  1. Um agente pode ter MÚLTIPLOS medidores (SMF). Somam-se as cargas por
     perfil de agente -> a variável-alvo é o consumo do PERFIL, não do medidor.
  2. O INMET publica em UTC. Converte-se para Brasília (UTC-3) ANTES do
     casamento temporal — sem isso, o clima entra 3 horas deslocado.
  3. As defasagens são calculadas DEPOIS da ordenação estrita por (agente,
     instante) e sobre uma grade horária COMPLETA. Calcular lag por deslocamento
     de POSIÇÃO numa série com lacunas produz desalinhamento silencioso — o erro
     que o script 01 existe para pegar.

Uso:  python scripts/00_build_dataset.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import glob, os

import numpy as np
import pandas as pd

from src import config as cfg

TZ_INMET = 3  # INMET em UTC -> Brasilia = UTC-3


# ---------------------------------------------------------------- CCEE
def ler_ccee(pasta=None) -> pd.DataFrame:
    """Lê os arquivos de medição/contabilização da CCEE e consolida por perfil."""
    pasta = pasta or cfg.RAW_CCEE
    arquivos = sorted(glob.glob(os.path.join(str(pasta), "*.csv")))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV da CCEE em {pasta}")

    partes = [pd.read_csv(f, sep=";", encoding="latin-1", low_memory=False) for f in arquivos]
    df = pd.concat(partes, ignore_index=True)
    df.columns = [c.strip().upper() for c in df.columns]

    df["DATA_DT"] = pd.to_datetime(df["DATA_DT"], errors="coerce")
    df["HORA"] = df["HORA"].astype(int)

    # (1) soma dos medidores por perfil de agente
    chave = ["SIGLA_PERFIL_AGENTE", "CODIGO_PERFIL_AGENTE", "DATA_DT", "HORA"]
    meta = ["CATEGORIA", "CLASSE", "PRODUTO", "OFERTA",
            "MONTANTE_PRELIMINAR_RD", "SINALIZADOR_HORARIO_ATEND_PROD"]
    agg = {"CONS_MW": "sum"}
    agg.update({c: "first" for c in meta if c in df.columns})
    return df.groupby(chave, as_index=False).agg(agg)


# ---------------------------------------------------------------- INMET
COLMAP_INMET = {
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temp_c",
    "UMIDADE RELATIVA DO AR, HORARIA (%)": "umid_rel",
    "RADIACAO GLOBAL (KJ/M²)": "rad_global",
    "PRECIPITAÇÃO TOTAL, HORÁRIO (MM)": "precip_mm",
    "VENTO, VELOCIDADE HORARIA (M/S)": "vento_vel",
}

def ler_inmet(pasta=None) -> pd.DataFrame:
    """Lê estações automáticas do INMET, converte UTC->UTC-3 e devolve por estação/hora."""
    pasta = pasta or cfg.RAW_INMET
    arquivos = sorted(glob.glob(os.path.join(str(pasta), "**", "*.CSV"), recursive=True))
    arquivos += sorted(glob.glob(os.path.join(str(pasta), "**", "*.csv"), recursive=True))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo do INMET em {pasta}")

    partes = []
    for f in arquivos:
        # os arquivos do INMET têm 8 linhas de cabeçalho de metadados da estação
        d = pd.read_csv(f, sep=";", decimal=",", encoding="latin-1", skiprows=8)
        d.columns = [c.strip().upper() for c in d.columns]
        d = d.rename(columns={k.upper(): v for k, v in COLMAP_INMET.items()})

        col_data = [c for c in d.columns if c.startswith("DATA")][0]
        col_hora = [c for c in d.columns if "HORA" in c and c not in COLMAP_INMET.values()][0]

        d["dt_utc"] = pd.to_datetime(
            d[col_data].astype(str).str.replace("/", "-", regex=False) + " "
            + d[col_hora].astype(str).str.extract(r"(\d{2})")[0] + ":00",
            errors="coerce",
        )
        # (2) UTC -> horario de Brasilia
        d["DATETIME"] = d["dt_utc"] - pd.Timedelta(hours=TZ_INMET)
        d["estacao"] = pathlib.Path(f).stem

        cols = [c for c in COLMAP_INMET.values() if c in d.columns]
        d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")
        d.loc[:, cols] = d[cols].mask(d[cols] <= -9999)   # sentinela de faltante do INMET
        partes.append(d[["estacao", "DATETIME"] + cols])

    inmet = pd.concat(partes, ignore_index=True).dropna(subset=["DATETIME"])
    inmet["DATA_DT"] = inmet["DATETIME"].dt.normalize()
    inmet["HORA"] = inmet["DATETIME"].dt.hour

    # interpolacao de lacunas curtas, preservando a tendencia
    cols = [c for c in COLMAP_INMET.values() if c in inmet.columns]
    inmet[cols] = (inmet.sort_values("DATETIME").groupby("estacao")[cols]
                   .transform(lambda s: s.interpolate(limit=3)))
    return inmet


# ---------------------------------------------------------------- calendário
def calendario(datas: pd.Series) -> pd.DataFrame:
    """dia_semana (0=seg) e feriado nacional."""
    d = pd.DataFrame({"DATA_DT": pd.to_datetime(datas.unique())})
    d["dia_semana"] = d["DATA_DT"].dt.dayofweek
    try:
        import holidays
        br = holidays.Brazil(years=range(2023, 2028))
        d["feriado"] = d["DATA_DT"].dt.date.map(lambda x: int(x in br))
    except ImportError:
        print("  [aviso] pacote `holidays` ausente -> feriado=0. Instale: pip install holidays")
        d["feriado"] = 0
    return d


# ---------------------------------------------------------------- defasagens
def adicionar_lags(df: pd.DataFrame) -> pd.DataFrame:
    """(3) lags sobre grade horária COMPLETA, após ordenação estrita."""
    df = df.sort_values(["SIGLA_PERFIL_AGENTE", "DATA_DT", "HORA"]).copy()
    df["DATETIME"] = df["DATA_DT"] + pd.to_timedelta(df["HORA"], unit="h")

    saida = []
    for ag, d in df.groupby("SIGLA_PERFIL_AGENTE"):
        grade = pd.date_range(d["DATETIME"].min(), d["DATETIME"].max(), freq="h")
        d = d.set_index("DATETIME").reindex(grade)     # expõe lacunas como NaN
        d["SIGLA_PERFIL_AGENTE"] = ag
        for k, nome in [(1, "lag_1h"), (24, "lag_24h"), (168, "lag_168h")]:
            d[nome] = d["CONS_MW"].shift(k)            # shift no TEMPO, não na posição
        saida.append(d.reset_index(names="DATETIME"))

    out = pd.concat(saida, ignore_index=True)
    out["DATA_DT"] = out["DATETIME"].dt.normalize()
    out["HORA"] = out["DATETIME"].dt.hour
    return out.drop(columns=["DATETIME"])


def main():
    print("1/4 CCEE...");   ccee = ler_ccee()
    print("2/4 INMET...");  clima = ler_inmet()

    print("3/4 Integrando...")
    # mapeie aqui cada agente à sua estação INMET de referência (proximidade da praça):
    #   ccee["estacao"] = ccee["SIGLA_PERFIL_AGENTE"].map(MAPA_AGENTE_ESTACAO)
    # e então: df = ccee.merge(clima, on=["estacao","DATA_DT","HORA"], how="left")
    raise SystemExit(
        "\nADAPTE: defina o mapeamento agente -> estacao INMET (praca de carga) e o\n"
        "merge correspondente antes de rodar. As funcoes de leitura acima ja estao\n"
        "prontas; falta apenas a chave de associacao geografica, que e especifica\n"
        "do seu conjunto de estacoes baixadas.\n"
    )

if __name__ == "__main__":
    main()
