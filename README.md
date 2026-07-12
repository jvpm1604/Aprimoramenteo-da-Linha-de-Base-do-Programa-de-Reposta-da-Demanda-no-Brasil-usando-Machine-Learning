.PHONY: all audit baseline acuracia financeiro figuras clean

all: audit baseline acuracia financeiro

audit:
	python scripts/01_auditoria_base.py

baseline:
	python scripts/02_features_e_baseline.py

acuracia: baseline
	python scripts/03_termossensibilidade.py
	python scripts/04_tabela_degrau.py
	python scripts/05_populacional.py
	python scripts/06_onda_calor.py

figuras: acuracia
	python scripts/08_truncamento_jensen.py
	python scripts/09_incerteza_ashrae.py

financeiro: baseline
	python scripts/07_financeiro.py

clean:
	rm -f outputs/tables/*.csv outputs/figures/*.png
	find . -name __pycache__ -type d -exec rm -rf {} +
