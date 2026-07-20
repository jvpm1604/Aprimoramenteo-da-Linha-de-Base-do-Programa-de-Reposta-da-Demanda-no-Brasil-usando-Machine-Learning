"""
Configuração central do projeto.

Todos os caminhos, constantes metodológicas e parâmetros regulatórios ficam
aqui — nenhum script deve conter "número mágico" hard-coded.

TCC: Aplicação de Algoritmos de Machine Learning para Determinação de Linhas
de Base no Âmbito do Sandbox Regulatório de Resposta da Demanda (UFRJ/DEE).
"""

from pathlib import Path

# ---------------------------------------------------------------- CAMINHOS
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

RAW_CCEE = DATA_RAW / "ccee"      # medição horária SMF por perfil de agente
RAW_INMET = DATA_RAW / "inmet"    # estações automáticas (um subdir por ano)
RAW_PLD = DATA_RAW / "pld"        # pld_horario_2024/2025/2026.csv

# Artefatos intermediários
DS_FINAL = DATA_INTERIM / "dataset_TCC_final_TCC.csv"        # saída do 01
DS_BASELINE = DATA_PROCESSED / "dataset_TCC_baseline_fiel.csv"  # saída do 03

OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"

# ------------------------------------------------------- ESCOPO TEMPORAL
PERIODO_INICIO = "2024-03-01"
PERIODO_FIM = "2026-01-31"
ANOS_PLD = [2024, 2025, 2026]

# --------------------------------------------- LINHA DE BASE REGULATÓRIA
# REN ANEEL 1.040/2022 + NT-ONS DGL 0049/2024.
# Defasagem de contabilização entre o mês de entrega e o mês de referência.
# A norma admite até 3 meses (até 2 a partir de 2024).
#
# ATENÇÃO: o valor abaixo é 1 (M-1) porque é o que reproduz o dataset original do
# TCC (100% de match). O texto do Capítulo 3 afirma M-2 — divergência a resolver.
# Ver docs/ARMADILHAS.md, Pendência 1. Rode `scripts/02 --lag 2` para a sensibilidade.
LAG_LB_MESES = 1

# Sábados: média dos sábados dos dois meses de referência anteriores.
SAB_MESES_REFERENCIA = [1, 2]

# Detecção de dias atípicos (critério MAD robusto)
MAD_K = 4.0                 # nº de desvios robustos para classificar outlier
MAD_CONST = 1.4826          # constante de consistência gaussiana do MAD
MIN_HORAS_DIA = 20          # horas válidas mínimas para o dia entrar na média
MIN_COBERTURA_MES = 0.50    # agente-mês com <50% de horas válidas é descartado

# ------------------------------------------------ AJUSTE INTRADIÁRIO
# Janela = 3 horas terminando 1 hora antes do início do evento (NESO/AEMO).
# Para evento iniciando em H0: W = {H0-4, H0-3, H0-2}; H0-1 é intervalo de guarda.
INDAY_JANELA_H = 3
INDAY_BUFFER_H = 1
INDAY_CAP = 0.20            # teto ±20% do fator multiplicativo (AEMO)

# ------------------------------------------------------ EVENTOS FICTÍCIOS
DUMMY_H0 = 14               # início da janela vespertina
DUMMY_H1 = 18               # fim (inclusive)

# --------------------------------------------------------- MODELOS DE ML
RANDOM_STATE = 42
RF_N_ESTIMATORS = 400
RF_MIN_SAMPLES_LEAF = 5
TEST_SIZE = 0.20            # split cronológico: 20% finais para teste

# Atributos
FEAT_CAL = ["hora_sin", "hora_cos", "dia_semana", "feriado", "is_weekend"]
FEAT_CLIMA = ["temp_c", "umid_rel", "rad_global", "precip_mm", "vento_vel", "CDD", "HDD"]
FEAT_LAG = ["lag_1h", "lag_24h", "lag_168h"]

FEAT_EXANTE = FEAT_CLIMA + FEAT_CAL             # RF ex ante (sem defasagens)
FEAT_COMPLETO = FEAT_CLIMA + FEAT_CAL + FEAT_LAG  # RF completo (teto de acurácia)

# TOWT: pontos de quebra (hinges) da resposta térmica linear por partes
TOWT_KNOTS_MIN = 12.0
TOWT_KNOTS_MAX = 40.0
TOWT_KNOTS_STEP = 3.0

# Graus-dia (base de conforto)
CDD_BASE = 18.0
HDD_BASE = 18.0

# ------------------------------------------------------------ FINANCEIRO
# Mapeamento agente -> submercado (para casar o PLD horário).
# ATENÇÃO: revisar/expandir com o cadastro oficial da CCEE.
SUBMERCADO_POR_AGENTE = {
    "NOVELIS": "SUDESTE",
    "ICB": "SUDESTE",
    "BOULEVARD RIO": "SUDESTE",
    "TOP SHOPPING": "SUDESTE",
    "BRASKEM SE ABC": "SUDESTE",
}
SUBMERCADO_DEFAULT = "SUDESTE"

N_BOOTSTRAP = 3000          # reamostragem dos dias de evento (IC 95%)

# Preço de oferta representativo para a valoração populacional (R$/MWh)
PRECO_OFERTA_REPRESENTATIVO = 700.0

# --------------------------------------------------------------- AGENTES
ARQUETIPOS = ["NOVELIS", "ICB", "BOULEVARD RIO", "TOP SHOPPING"]

# Excluído dos resultados por quebra de regime na série (documentado no TCC).
AGENTES_EXCLUIDOS = ["BRASKEM SE ABC"]

# ---------------------------------------------------------- INCERTEZA M&V
ASHRAE_CONFIANCA = 0.90
ASHRAE_N = 1000             # pontos do período de linha de base
ASHRAE_M = 40               # pontos do período de apuração
