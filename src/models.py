"""
Modelos de Machine Learning usados como REFERÊNCIA DE ACURÁCIA (teto), não como
proposta operacional.

  - RandomForest ex ante  : calendário + clima, SEM defasagens.
        Mesma classe de informação da linha de base publicada -> comparador justo.
  - RandomForest completo : + defasagens (1h, 24h, 168h).
        Teto de acurácia quando há acesso à recência da carga.
  - TOWT (Time-Of-Week and Temperature, CalTRACK/CAISO):
        168 indicadores de hora-da-semana + resposta térmica LINEAR POR PARTES.
        Diferentemente das árvores, EXTRAPOLA além da faixa de treino — decisivo
        em ondas de calor, onde o RF satura no máximo aprendido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from . import config as cfg


def novo_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=cfg.RF_N_ESTIMATORS,
        min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1,
    )


def treinar_rf(treino: pd.DataFrame, feats: list[str], alvo: str = "CONS_MW"):
    tr = treino.dropna(subset=feats + [alvo])
    if len(tr) < 100:
        return None
    return novo_rf().fit(tr[feats], tr[alvo])


# --------------------------------------------------------------------------
KNOTS = np.arange(cfg.TOWT_KNOTS_MIN, cfg.TOWT_KNOTS_MAX, cfg.TOWT_KNOTS_STEP)


class TOWT:
    """Time-Of-Week and Temperature (CalTRACK 2.0 Hourly, adotado pelo CAISO).

    X = [168 dummies de hora-da-semana | temp | hinges max(0, temp - knot) | umidade]

    As funções-rampa (hinges) dão não linearidade térmica preservando a
    extrapolação LINEAR fora da faixa de treino.
    """

    COLS = ["temp_c", "umid_rel", "dia_semana", "HORA"]

    @staticmethod
    def _X(d: pd.DataFrame) -> np.ndarray:
        tow = (d["dia_semana"].astype(int) * 24 + d["HORA"].astype(int)).values
        TOW = pd.get_dummies(pd.Categorical(tow, categories=range(168)),
                             dtype=float).values
        t = d["temp_c"].values.reshape(-1, 1)
        hinges = np.maximum(0.0, t - KNOTS.reshape(1, -1))
        hum = d["umid_rel"].values.reshape(-1, 1)
        return np.hstack([TOW, t, hinges, hum])

    def fit(self, d: pd.DataFrame, alvo: str = "CONS_MW"):
        m = d[self.COLS + [alvo]].notna().all(axis=1)
        d = d[m]
        self.lr = LinearRegression().fit(self._X(d), d[alvo].values)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self.lr.predict(self._X(d))


# --------------------------------------------------------------------------
def prever_recursivo(modelo, evento: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Previsão em horas de evento CONSECUTIVAS, com recursão do lag_1h.

    SALVAGUARDA CONTRA VAZAMENTO: dentro de um evento de múltiplas horas, o
    lag_1h medido já está REDUZIDO (é o consumo suprimido da hora anterior).
    Usá-lo rebaixaria artificialmente a linha de base e subestimaria a redução.
    Aqui, o lag_1h é realimentado pela PRÓPRIA previsão contrafactual.

    Aviso: lag_24h e lag_168h também podem cair em horas de evento quando há
    despachos em dias seguidos (caso do ICB). Por isso o modelo completo NÃO é
    usado na trilha financeira — apenas em eventos fictícios, onde os lags são
    limpos por construção. Ver docs/ARMADILHAS.md.
    """
    ev = evento.sort_values("HORA").copy()
    preds, anterior = [], {}
    for _, r in ev.iterrows():
        x = r[feats].copy()
        h = int(r["HORA"])
        if "lag_1h" in feats and (h - 1) in anterior:
            x["lag_1h"] = anterior[h - 1]
        X = pd.DataFrame([x], columns=feats).astype(float)
        p = float(modelo.predict(X.fillna(0.0))[0])
        anterior[h] = p
        preds.append(p)
    return np.asarray(preds)
