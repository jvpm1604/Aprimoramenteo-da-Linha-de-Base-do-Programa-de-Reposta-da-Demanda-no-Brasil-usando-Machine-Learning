"""
Carregamento do dataset base e engenharia de atributos.

O arquivo `dataset_TCC_final_TCC.csv` (saída do script 00) traz medição CCEE +
clima INMET + metadados de RD + defasagens. Este módulo acrescenta os atributos
derivados exigidos pelos modelos e pela reconstrução da linha de base:

    DATETIME, data, ym, is_util, is_sab, hora_sin, hora_cos, is_weekend,
    CDD, HDD, OFERTA_num, evento (bool), fonte_valida, treino_valido

Convenção de `dia_semana`: 0 = segunda ... 6 = domingo (padrão pandas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

SEP = ";"  # o CSV da CCEE é exportado com ponto e vírgula


# --------------------------------------------------------------------------
def carregar_base(path=None) -> pd.DataFrame:
    """Lê o dataset consolidado bruto (saída do script 00)."""
    path = path or cfg.DS_FINAL
    df = pd.read_csv(path, sep=SEP, low_memory=False)
    df["DATA_DT"] = pd.to_datetime(df["DATA_DT"], errors="coerce")
    df["HORA"] = df["HORA"].astype(int)
    return df


def adicionar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Cria todos os atributos derivados. Idempotente."""
    df = df.copy()

    # --- tempo -----------------------------------------------------------
    df["data"] = df["DATA_DT"].dt.normalize()
    df["DATETIME"] = df["data"] + pd.to_timedelta(df["HORA"], unit="h")
    # ym = AAAAMM (rótulo do mês, usado na linha de base mensal)
    df["ym"] = df["data"].dt.year * 100 + df["data"].dt.month

    # --- tipo de dia (grade regulatória: útil / sábado) -------------------
    feriado = df["feriado"].astype(int) == 1
    df["is_util"] = (df["dia_semana"] <= 4) & (~feriado)   # seg-sex, não feriado
    df["is_sab"] = (df["dia_semana"] == 5) & (~feriado)
    df["is_weekend"] = (df["dia_semana"] >= 5).astype(int)
    # domingos e feriados ficam fora da grade -> linha de base indefinida

    # --- ciclicidade horária ---------------------------------------------
    df["hora_sin"] = np.sin(2 * np.pi * df["HORA"] / 24.0)
    df["hora_cos"] = np.cos(2 * np.pi * df["HORA"] / 24.0)

    # --- graus-dia --------------------------------------------------------
    df["CDD"] = np.maximum(0.0, df["temp_c"] - cfg.CDD_BASE)
    df["HDD"] = np.maximum(0.0, cfg.HDD_BASE - df["temp_c"])

    # --- metadados de RD --------------------------------------------------
    # OFERTA vem como texto ("NAO_PARTICIPANTE" ou o preço em R$/MWh)
    df["OFERTA_num"] = pd.to_numeric(df["OFERTA"], errors="coerce")
    df["evento"] = df["SINALIZADOR_HORARIO_ATEND_PROD"].fillna(0).astype(int) > 0

    # dias com QUALQUER hora de despacho -> blackout (dia inteiro)
    dias_evento = (
        df.loc[df["evento"], ["SIGLA_PERFIL_AGENTE", "data"]]
        .drop_duplicates()
        .assign(dia_evento=True)
    )
    df = df.merge(dias_evento, on=["SIGLA_PERFIL_AGENTE", "data"], how="left")
    df["dia_evento"] = df["dia_evento"].fillna(False)

    df = limpar_consumo(df)

    # --- máscaras de uso --------------------------------------------------
    # fonte_valida: a hora pode compor a MÉDIA da linha de base
    df["fonte_valida"] = df["CONS_MW"].notna() & (~df["dia_evento"])
    # treino_valido: a hora pode entrar no TREINO dos modelos de ML
    df["treino_valido"] = df["fonte_valida"]

    return df.sort_values(["SIGLA_PERFIL_AGENTE", "DATETIME"]).reset_index(drop=True)


def limpar_consumo(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processamento do consumo (critérios de conformidade tipo CAISO).

    1. Consumo RIGOROSAMENTE NULO -> NaN (manutenção programada ou falha de
       comunicação do SMF). Não se aplica limiar relativo: cargas legitimamente
       pequenas (agentes com média ~1 MW, como os shoppings) seriam apagadas.
       Esta é a regra do pipeline original — validada por reprodução exata.
    2. Agente-mês com cobertura < 50% de horas válidas -> descartado das médias.
    """
    df = df.copy()
    espurio = df["CONS_MW"] == 0
    df.loc[espurio, "CONS_MW"] = np.nan
    df["cons_espurio"] = espurio

    # cobertura mensal
    cob = (
        df.assign(ok=df["CONS_MW"].notna())
        .groupby(["SIGLA_PERFIL_AGENTE", "ym"])["ok"]
        .mean()
        .rename("cobertura_mes")
        .reset_index()
    )
    df = df.merge(cob, on=["SIGLA_PERFIL_AGENTE", "ym"], how="left")
    df["mes_valido"] = df["cobertura_mes"] >= cfg.MIN_COBERTURA_MES
    return df


def preparar(path=None) -> pd.DataFrame:
    """Atalho: carrega + adiciona atributos."""
    return adicionar_atributos(carregar_base(path))


# --------------------------------------------------------------------------
def split_cronologico(d: pd.DataFrame, test_size: float = None):
    """Divisão cronológica por agente: primeiros (1-test) para treino.

    Evita o vazamento da validação cruzada aleatória em séries temporais.
    Retorna (treino, teste). O treino já é filtrado por `treino_valido`.
    """
    test_size = cfg.TEST_SIZE if test_size is None else test_size
    corte = d["DATETIME"].quantile(1 - test_size)
    tr = d[(d["DATETIME"] < corte) & (d["treino_valido"])]
    te = d[d["DATETIME"] >= corte]
    return tr, te
