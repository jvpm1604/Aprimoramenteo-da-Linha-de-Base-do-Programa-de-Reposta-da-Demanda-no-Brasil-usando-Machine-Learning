"""
Script 00 — Consolidação da base horária (produz dataset_TCC_final_TCC.csv) que é o insumo exclusivo do projeto.
Nesse arquivo vou unir todos os dados necessários para análise da baseline atual, influência da Temperatura,
e proposta de melhoria para criação de nova metodologia de Baseline.

  medição CCEE (SMF)  +  clima INMET  +  calendário  ->  uma linha por agente-hora

Obs: este script é uma reconstrução documentada do pipeline original (que
rodava no Google Colab sobre o Drive). Por via das dúvidas, deixarei aqui os links de Dados Abertos usados para facilitar
o entendimento e possível interesse de reconstrução por parte do Leitor, destacando que A ESTRUTURA E ATÉ CAMINHO (LINK)
PARA OS DADOS PODE ALTERAR, CONSIDERANDO O MOMENTO QUE A CONSULTA SERÁ REALIZADA:

Consumo horário dos Agente: https://dadosabertos.ccee.org.br/dataset/consumo_horario_perfil_agente
Dados metereológicos Brasileiros: https://portal.inmet.gov.br/dadoshistoricos
Dados relativos ao Sandbox de RD: https://dadosabertos.ccee.org.br/dataset/rd_disp_despacho_horario_sandbox

Ajuste `ler_ccee` e `ler_inmet` conforme necessário. O contrato de SAÍDA, esse sim, é fixo — é o que
os scripts 01+ consomem:

    SIGLA_PERFIL_AGENTE; DATA_DT; HORA; CONS_MW;
    temp_c; umid_rel; rad_global; precip_mm; vento_vel;
    CATEGORIA; CLASSE; PRODUTO; OFERTA; MONTANTE_PRELIMINAR_RD;
    SINALIZADOR_HORARIO_ATEND_PROD; dia_semana; feriado;
    DATA_STR; CODIGO_PERFIL_AGENTE; lag_1h; lag_24h; lag_168h

DECISÕES QUE IMPORTAM (e por quê(?)):
  1. Um agente pode ter MÚLTIPLOS medidores (SMF). Somam-se as cargas por
     perfil de agente -> a variável-alvo é o consumo do PERFIL, não do medidor.
  2. O INMET publica em UTC. Converte-se para Brasília (UTC-3) ANTES do
     casamento temporal — sem isso, o clima entra 3 horas deslocado.
  3. As defasagens são calculadas DEPOIS da ordenação estrita por (agente,
     instante) e sobre uma grade horária COMPLETA. Calcular lag por deslocamento
     de POSIÇÃO numa série com lacunas produz desalinhamento silencioso — o erro
     que o script 01 existe para pegar.
  4. TEMPERATURA (INMET) tem duas limpezas antes de virar dado utilizável:
     (a) o INMET usa -9999 como sinonimo de "sem leitura", e sem mascarar isso
         antes de qualquer conta, -9999°C entra como se fosse um valor real e
         destrói qualquer média/correlação que toque a série;
     (b) lacunas CURTAS (até 3 horas) são preenchidas por interpolação linear,
         que preserva a tendência térmica local; lacunas mais longas que isso
         permanecem como NaN de propósito — não se inventa clima para além do
         que é razoável interpolar. Ambas as limpezas rodam por ESTAÇÃO,
         nunca misturando a série de uma estação com a de outra.
  5. Um agente pode ter MÚLTIPLAS OFERTAS de RD na mesma hora (lotes
     empilhados por ordem de mérito — comportamento normal do mercado, não
     erro). A consolidação por agente-hora precisa SOMAR o montante entre as
     ofertas e tomar o MÁXIMO do sinalizador de despacho (1 se qualquer oferta
     daquela hora foi confirmada). Usar "pegue a primeira" (`first`) aqui é um
     bug real, não uma simplificação: ele pode descartar justamente a oferta
     que tinha o despacho confirmado, fazendo um dia de evento real parecer,
     para o resto do pipeline, um dia comum — contaminando silenciosamente o
     cálculo da linha de base com consumo que já estava reduzido.

Uso:  python scripts/00_build_dataset.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import glob, os

import numpy as np
import pandas as pd

from src import config as cfg

TZ_INMET = 3  # INMET em UTC -> Brasilia = UTC-3


# ----- CCEE (base de código para direcionamento do Leitor)
def ler_ccee(pasta=None) -> pd.DataFrame:
    """Lê os arquivos de medição/contabilização da CCEE e consolida por perfil.

    Consolidação em dois níveis, cada um com sua regra própria (ver decisões
    1 e 5 no cabeçalho):
      - CONS_MW            -> SOMA (múltiplos medidores do mesmo agente)
      - MONTANTE_PRELIMINAR_RD -> SOMA (múltiplas ofertas na mesma hora)
      - SINALIZADOR_HORARIO_ATEND_PROD -> MÁXIMO (1 se QUALQUER oferta confirmou despacho)
      - demais metadados (CATEGORIA, CLASSE, PRODUTO, OFERTA) -> primeira
        ocorrência, por serem estáveis dentro do agente-hora e não terem
        uma regra de agregação numérica óbvia.
    """
    pasta = pasta or cfg.RAW_CCEE
    arquivos = sorted(glob.glob(os.path.join(str(pasta), "*.csv")))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV da CCEE em {pasta}")

    partes = [pd.read_csv(f, sep=";", encoding="latin-1", low_memory=False) for f in arquivos]
    df = pd.concat(partes, ignore_index=True)
    df.columns = [c.strip().upper() for c in df.columns]

    df["DATA_DT"] = pd.to_datetime(df["DATA_DT"], errors="coerce")
    df["HORA"] = df["HORA"].astype(int)

    if "MONTANTE_PRELIMINAR_RD" in df.columns:
        df["MONTANTE_PRELIMINAR_RD"] = pd.to_numeric(
            df["MONTANTE_PRELIMINAR_RD"], errors="coerce"
        ).fillna(0.0)
    if "SINALIZADOR_HORARIO_ATEND_PROD" in df.columns:
        df["SINALIZADOR_HORARIO_ATEND_PROD"] = pd.to_numeric(
            df["SINALIZADOR_HORARIO_ATEND_PROD"], errors="coerce"
        ).fillna(0).astype(int)

    # (1) soma dos medidores por perfil de agente
    chave = ["SIGLA_PERFIL_AGENTE", "CODIGO_PERFIL_AGENTE", "DATA_DT", "HORA"]

    agg = {"CONS_MW": "sum"}
    if "MONTANTE_PRELIMINAR_RD" in df.columns:
        agg["MONTANTE_PRELIMINAR_RD"] = "sum"      # (5) soma entre ofertas simultâneas
    if "SINALIZADOR_HORARIO_ATEND_PROD" in df.columns:
        agg["SINALIZADOR_HORARIO_ATEND_PROD"] = "max"  # (5) 1 se QUALQUER oferta despachou

    meta_estavel = ["CATEGORIA", "CLASSE", "PRODUTO", "OFERTA"]
    agg.update({c: "first" for c in meta_estavel if c in df.columns})

    n_antes = len(df)
    consolidado = df.groupby(chave, as_index=False).agg(agg)
    n_colisoes = int((df.groupby(chave).size() > 1).sum())
    print(f"  [ler_ccee] {n_antes} linhas brutas -> {len(consolidado)} agente-hora "
          f"({n_colisoes} horas tinham mais de uma oferta/medidor, agregadas por soma+max)")

    return consolidado


# ---- INMET
COLMAP_INMET = {
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)": "temp_c",
    "UMIDADE RELATIVA DO AR, HORARIA (%)": "umid_rel",
    "RADIACAO GLOBAL (KJ/M²)": "rad_global",
    "PRECIPITAÇÃO TOTAL, HORÁRIO (MM)": "precip_mm",
    "VENTO, VELOCIDADE HORARIA (M/S)": "vento_vel",
}

def ler_inmet(pasta=None) -> pd.DataFrame:
    """Lê estações automáticas do INMET, converte UTC -> UTC-3 e devolve por estação/hora.

    Duas limpezas acontecem aqui (ver decisão 4 no cabeçalho):
      - sentinela -9999 (código do INMET para "sem leitura") vira NaN;
      - lacunas de até 3 horas são interpoladas linearmente, por estação.
    """
    pasta = pasta or cfg.RAW_INMET
    arquivos = sorted(glob.glob(os.path.join(str(pasta), "**", "*.CSV"), recursive=True))
    arquivos += sorted(glob.glob(os.path.join(str(pasta), "**", "*.csv"), recursive=True))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo do INMET em {pasta}!")

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
        # (4a) sentinela de faltante do INMET -> NaN (senão -9999°C entra como dado real)
        d.loc[:, cols] = d[cols].mask(d[cols] <= -9999)
        partes.append(d[["estacao", "DATETIME"] + cols])

    inmet = pd.concat(partes, ignore_index=True).dropna(subset=["DATETIME"])
    inmet["DATA_DT"] = inmet["DATETIME"].dt.normalize()
    inmet["HORA"] = inmet["DATETIME"].dt.hour

    # (4b) interpolacao de lacunas curtas (<=3h), preservando a tendencia, por estacao
    # (nunca entre estacoes diferentes: o groupby garante isso).
    cols = [c for c in COLMAP_INMET.values() if c in inmet.columns]
    inmet[cols] = (inmet.sort_values("DATETIME").groupby("estacao")[cols]
                   .transform(lambda s: s.interpolate(limit=3)))
    return inmet


# ---------------------------------------------------------------- calendário
def calendario(datas: pd.Series) -> pd.DataFrame:
    """dia_semana (0=seg) e feriado nacional.

    ATENÇÃO: sem o pacote `holidays`, feriados NÃO são excluídos do cálculo da
    linha de base — e feriados costumam ter consumo bem menor que um dia útil
    comum, então essa omissão contaminaria a média silenciosamente. Por isso,
    a ausência do pacote é tratada como ERRO, não como aviso ignorável.
    """
    d = pd.DataFrame({"DATA_DT": pd.to_datetime(datas.unique())})
    d["dia_semana"] = d["DATA_DT"].dt.dayofweek
    try:
        import holidays
    except ImportError as e:
        raise ImportError(
            "Pacote `holidays` ausente. Sem ele, feriados nacionais NÃO seriam "
            "excluídos do cálculo da linha de base, contaminando a média com dias "
            "de consumo atipicamente baixo. Instale antes de continuar: "
            "pip install holidays"
        ) from e
    br = holidays.Brazil(years=range(2023, 2028))
    d["feriado"] = d["DATA_DT"].dt.date.map(lambda x: int(x in br))
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
