# Linhas de Base para Resposta da Demanda — código do TCC

Código do Trabalho de Conclusão de Curso **"Aplicação de Algoritmos de Machine Learning para
Determinação de Linhas de Base no Âmbito do Sandbox Regulatório de Resposta da Demanda"**
(João Vitor Pires Marinho — UFRJ / Escola Politécnica / DEE, 2026).

## A tese, em uma frase

A linha de base da CCEE — média horária mensal publicada *ex ante* — corresponde à etapa
**não ajustada** das metodologias de mercados maduros (NESO, AEMO, ISO-NE). Falta-lhe o
**ajuste intradiário**, que corrige a média ao nível efetivo do dia usando a medição das horas
anteriores ao despacho. Este trabalho mostra, com dados de 43 agentes, que essa correção —
simples, auditável e computável com a medição já existente — captura a maior parte do ganho de
acurácia atingível, e que **modelos de Machine Learning servem como teto de referência, não como
proposta operacional**.

## A comparação em três degraus

| Degrau | Estimador | Papel |
|---|---|---|
| (i) | Linha de base estática (CCEE) | *status quo*, reconstruído fielmente à norma |
| (ii) | + ajuste intradiário — aditivo (NESO/ISO-NE) e multiplicativo ±20% (AEMO) | **a proposta** |
| (iii) | RF *ex ante*, RF completo, TOWT | teto de acurácia e diagnóstico da física da carga |

Comparar (ii) com (i) mede o efeito do ajuste. Comparar (iii) com os demais mede quanto ainda
seria recuperável por um modelo complexo — e, portanto, se o ajuste simples basta.

## Estrutura

```
src/                    módulos reutilizáveis (a lógica vive aqui)
  config.py             TODOS os parâmetros: caminhos, regras, hiperparâmetros
  features.py           carregamento + engenharia de atributos + split cronológico
  baseline.py           reconstrução FIEL da linha de base da CCEE
  inday.py              ajuste intradiário (aditivo e multiplicativo)
  models.py             Random Forest e TOWT (+ recursão de lag anti-vazamento)
  dummy_events.py       motor dos eventos fictícios (a verdade de campo)
  metrics.py            NMBE, CV(RMSE), MAPE
  financeiro.py         PLD, apuração truncada, ESS, bootstrap
  truncamento.py        viés de Jensen + incerteza ASHRAE 14

scripts/                pipeline executável, em ordem
  00_build_dataset.py       CCEE + INMET + calendário -> dataset_TCC_final_TCC.csv
  01_auditoria_base.py      integridade: contiguidade, unicidade, alinhamento de lags
  02_features_e_baseline.py atributos + linha de base fiel -> dataset_TCC_baseline_fiel.csv
  03_termossensibilidade.py correlação consumo x temperatura (seleção de arquétipos)
  04_tabela_degrau.py       6 estimadores nos arquétipos      [Tabelas 4.1 e 4.2]
  05_populacional.py        43 agentes + boxplot              [Seção 4.3, Apêndice A]
  06_onda_calor.py          Trilha B: extrapolação térmica    [Tabela 4.5]
  07_financeiro.py          eventos reais, PLD, bootstrap     [Tabela 4.6]
  08_truncamento_jensen.py  Monte Carlo + figura de Jensen    [Seção 4.5]
  09_incerteza_ashrae.py    incerteza da economia             [Tabela 4.7]

data/raw/               dados brutos (NÃO versionados)
  ccee/  inmet/  pld/
data/interim/           dataset_TCC_final_TCC.csv
data/processed/         dataset_TCC_baseline_fiel.csv
outputs/tables/         CSVs que alimentam as tabelas do TCC
outputs/figures/        figuras do TCC
docs/                   MAPA_TCC.md (código -> tabela/figura) e ARMADILHAS.md
```

## Como rodar

```bash
pip install -r requirements.txt

# coloque dataset_TCC_final_TCC.csv em data/interim/
# (ou rode o 00 a partir dos dados brutos da CCEE/INMET)

python scripts/01_auditoria_base.py        # audita ANTES de qualquer análise
python scripts/02_features_e_baseline.py   # gera a linha de base fiel
python scripts/03_termossensibilidade.py
python scripts/04_tabela_degrau.py
python scripts/05_populacional.py          # --ml para incluir RF/TOWT nos 43
python scripts/06_onda_calor.py
python scripts/07_financeiro.py            # requer os PLDs em data/raw/pld/
python scripts/08_truncamento_jensen.py    # requer o 05
python scripts/09_incerteza_ashrae.py      # requer o 05
```

Sensibilidade da defasagem de contabilização:
`python scripts/02_features_e_baseline.py --lag 1` (a norma admite até 2 meses; caso base = 2).

## Resultados-chave que o pipeline reproduz

- Ajuste aditivo reduz o CV(RMSE) em **42 dos 43 agentes**; mediana **13,3%**.
- Crédito espúrio mediano cai **53%** (para 4,6% da carga) — reduzindo a superavaliação
  socializada via ESS.
- Monte Carlo confirma a constante de Jensen: **0,398σ** (teoria: 0,399σ).
- Nas cargas comerciais, o ajuste iguala ou supera o teto de ML; nas industriais, o RF completo
  mantém vantagem, ao custo da auditabilidade.

## Aviso importante

O pipeline reproduz o `dataset_TCC_baseline_fiel.csv` original com **100% de match** e todos os números do TCC. Restam **duas pendências no TEXTO** (defasagem M−1 vs M−2; alinhamento de linhas na Tabela 4.5). Ver **`docs/ARMADILHAS.md`**.
