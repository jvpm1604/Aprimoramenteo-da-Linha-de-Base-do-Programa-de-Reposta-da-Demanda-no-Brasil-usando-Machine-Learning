# Mapa: código ↔ TCC

Rastreabilidade de cada número do texto até o script que o produz.

## Capítulo 3 — Metodologia

| Item do TCC | Onde está no código |
|---|---|
| §3.2 Fontes de dados e consolidação | `scripts/00_build_dataset.py` |
| §3.3 Validação da integridade da base | `scripts/01_auditoria_base.py` |
| Tabela 3.2 — Termossensibilidade | `scripts/03_termossensibilidade.py` |
| §3.5 Pré-processamento e limpeza | `src/features.py::limpar_consumo` |
| §3.5 Janela de *blackout* | `src/features.py` (coluna `dia_evento`) |
| Eq. 3.1 — Linha de base regulatória | `src/baseline.py::calcular_baseline_ccee` |
| §3.6 Detecção de dias atípicos (MAD) | `src/baseline.py::marcar_dias_atipicos` |
| Eq. 3.3 — Ajuste aditivo | `src/inday.py::offset_aditivo` |
| Eq. 3.4 — Ajuste multiplicativo (±20%) | `src/inday.py::fator_multiplicativo` |
| Tabela 3.4 — Atributos de entrada | `src/config.py` (`FEAT_*`) |
| §3.8 Random Forest | `src/models.py::novo_rf` |
| §3.8 TOWT | `src/models.py::TOWT` |
| §3.8 Validação (split cronológico) | `src/features.py::split_cronologico` |
| §3.9 Métricas (NMBE, CV(RMSE), MAPE) | `src/metrics.py` |
| §3.10.1 Eventos fictícios | `src/dummy_events.py` |
| §3.10.2 Trilha financeira | `src/financeiro.py`, `scripts/07` |
| §3.11 Incerteza (ASHRAE 14) | `src/truncamento.py::incerteza_ashrae` |
| §3.12 Formalização do viés do truncamento | `src/truncamento.py` |
| §3.13 Tratamento de vazamento | `src/models.py::prever_recursivo` + `dia_evento` + janela pré-evento |

## Capítulo 4 — Resultados

| Item do TCC | Script | Saída |
|---|---|---|
| Tabela 4.1 — CV(RMSE) por estimador | `04_tabela_degrau.py` | `outputs/tables/04_degrau_cvrmse.csv` |
| Tabela 4.2 — NMBE por estimador | `04_tabela_degrau.py` | `outputs/tables/04_degrau_nmbe.csv` |
| Tabela 4.3 — Irrelevância do clima | `04` (rodar variando `FEAT_*`) | — *ver nota abaixo* |
| Figura 4.1 — Boxplot populacional | `05_populacional.py` | `outputs/figures/fig_boxplot_populacional.png` |
| §4.3 — Medianas dos 43 agentes | `05_populacional.py` | console + `05_populacional_por_agente.csv` |
| Tabela 4.5 — Onda de calor (Trilha B) | `06_onda_calor.py` | `outputs/tables/06_onda_calor.csv` |
| §4.5 + Figura 4.2 — Jensen | `08_truncamento_jensen.py` | `outputs/figures/fig_truncamento_jensen.png` |
| Tabela 4.6 — Distorção financeira | `07_financeiro.py` | `outputs/tables/07_financeiro_por_agente.csv` |
| Tabela 4.7 — Incerteza ASHRAE | `09_incerteza_ashrae.py` | `outputs/tables/09_incerteza_ashrae.csv` |

**Nota sobre a Tabela 4.3.** Ela compara o RF em três configurações de atributos
(calendário+lag / completo / calendário+clima). Ainda não há script dedicado; é a única
tabela do Capítulo 4 sem um caminho de reprodução direto. Sugestão: criar
`scripts/04b_ablacao_clima.py` reaproveitando `src/dummy_events.avaliar_agente` com listas de
atributos alternativas.

## Apêndices

| Item | Script | Saída |
|---|---|---|
| Apêndice A — Tabela por agente (43 linhas) | `05_populacional.py` | `05_populacional_por_agente.csv` |
| Apêndice B — Listagens de código | `src/baseline.py`, `src/inday.py`, `src/truncamento.py` |

O Apêndice B do TCC reproduz trechos condensados. Eles correspondem, respectivamente, a
`calcular_baseline_ccee`, `inday.aplicar` e `truncamento.monte_carlo`.
