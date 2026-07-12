"""
Script 05 — Validação populacional nos 43 agentes (Seção 4.3 + Apêndice A + boxplot).

Por que populacional: estimativas por agente são frágeis (poucos eventos, ruído
alto). Sobre a totalidade dos agentes, o ruído individual se cancela e sobra o
sinal sistemático. É o método de avaliação em larga escala do LBNL.

Calcula, por agente, em eventos fictícios (verdade de campo -> redução real = 0):
  - CV(RMSE) e NMBE sob a linha de base estática e sob os dois ajustes;
  - CRÉDITO ESPÚRIO = media(max(0, B - y)) / media(y), em % da carga.
    Como a redução verdadeira é NULA, todo esse volume é erro puro.

Saídas: tabela por agente (Apêndice A) + fig_boxplot_populacional.png

Uso:  python scripts/05_populacional.py            (sem ML, rápido)
      python scripts/05_populacional.py --ml       (com RF/TOWT, lento)
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as cfg
from src.dummy_events import avaliar_agente
from src.metrics import cvrmse, nmbe
from src.truncamento import credito_espurio_empirico

ROTULOS = {"1_estatica": "Estatica (CCEE)", "2_aditivo": "Aditivo", "3_multiplicativo": "Multiplicativo"}

def main(com_ml: bool):
    df = pd.read_csv(cfg.DS_BASELINE, low_memory=False)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"]); df["data"] = pd.to_datetime(df["data"])

    linhas = []
    for ag, d in df.groupby("SIGLA_PERFIL_AGENTE"):
        r = avaliar_agente(d, com_ml=com_ml)
        if r is None:
            print(f"  [pulado] {ag}"); continue
        row = dict(agente=ag, dias=r["n_dias"], obs=r["n_obs"])
        for k, p in r["pred"].items():
            row[f"CVR_{k}"] = round(cvrmse(r["y"], p), 2)
            row[f"NMBE_{k}"] = round(nmbe(r["y"], p), 2)
            row[f"SP_{k}"] = round(credito_espurio_empirico(p, r["y"]), 2)
        linhas.append(row)

    t = pd.DataFrame(linhas).sort_values("CVR_1_estatica", ascending=False)
    t.to_csv(cfg.OUT_TABLES / "05_populacional_por_agente.csv", index=False)

    # ---------------- síntese ----------------
    n = len(t)
    print(f"\n{'='*64}\nPOPULACIONAL — {n} agentes\n{'='*64}")
    for met, nome in [("CVR", "CV(RMSE)"), ("SP", "Credito espurio (% carga)")]:
        e = t[f"{met}_1_estatica"].median(); a = t[f"{met}_2_aditivo"].median()
        m = t[f"{met}_3_multiplicativo"].median()
        print(f"{nome:28s} mediana: estatica={e:6.2f} | aditivo={a:6.2f} "
              f"({100*(a-e)/e:+.0f}%) | mult={m:6.2f}")
    vb_e = t["NMBE_1_estatica"].abs().median(); vb_a = t["NMBE_2_aditivo"].abs().median()
    print(f"{'|NMBE| mediano':28s}        : estatica={vb_e:6.2f} | aditivo={vb_a:6.2f}")

    melhora_cv = int((t["CVR_2_aditivo"] < t["CVR_1_estatica"]).sum())
    melhora_vb = int((t["NMBE_2_aditivo"].abs() < t["NMBE_1_estatica"].abs()).sum())
    print(f"\nAditivo melhora o CV(RMSE) em {melhora_cv}/{n} agentes")
    print(f"Aditivo melhora o |vies|   em {melhora_vb}/{n} agentes")

    # ---------------- boxplot ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (met, nome) in zip(axes, [("CVR", "CV(RMSE) (%)"), ("NMBE", "NMBE (%)")]):
        dados = [t[f"{met}_{k}"].dropna() for k in ROTULOS]
        bp = ax.boxplot(dados, tick_labels=list(ROTULOS.values()), showmeans=True, patch_artist=True)
        for p, c in zip(bp["boxes"], ["#c0392b", "#27ae60", "#2980b9"]):
            p.set_facecolor(c); p.set_alpha(0.45)
        ax.set_title(f"{nome} entre os {n} agentes"); ax.grid(alpha=0.3, axis="y")
        if met == "NMBE": ax.axhline(0, color="k", lw=0.8, ls="--")
    fig.suptitle("Eventos ficticios (14h-18h): distribuicao do erro por estimador")
    fig.tight_layout()
    f = cfg.OUT_FIGURES / "fig_boxplot_populacional.png"
    fig.savefig(f, dpi=200); print(f"\n-> {f}")
    print(f"-> {cfg.OUT_TABLES}/05_populacional_por_agente.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml", action="store_true", help="inclui RF e TOWT (lento)")
    main(ap.parse_args().ml)
