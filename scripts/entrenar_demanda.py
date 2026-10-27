#!/usr/bin/env python3
"""Entrena un Random Forest de demanda con series sintéticas y guarda el joblib.

Uso (desde ms-ia):
  python scripts/entrenar_demanda.py
  python scripts/entrenar_demanda.py --salida modelo/modelo_demanda.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.demanda import FEATURES, VENTANA, features_de  # noqa: E402


def serie_sintetica(rng: np.random.Generator) -> tuple[list[float], float]:
    """Serie de 90 días + demanda objetivo de los próximos 30 (no observada)."""
    tasa = float(rng.choice([0.05, 0.15, 0.35, 0.7, 1.2, 2.0]))
    serie = rng.poisson(tasa, size=VENTANA).astype(float)
    # Picos ocasionales
    if rng.random() < 0.3:
        idx = rng.integers(0, VENTANA, size=3)
        serie[idx] += rng.integers(1, 5, size=3)
    futuro = rng.poisson(tasa, size=30).astype(float)
    demanda30 = float(futuro.sum())
    # Ruido ligero en la etiqueta
    demanda30 = max(0.0, demanda30 + float(rng.normal(0, 0.5)))
    return serie.tolist(), demanda30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--salida",
        default=str(ROOT / "modelo" / "modelo_demanda.joblib"),
    )
    parser.add_argument("--muestras", type=int, default=4000)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.semilla)
    filas = []
    etiquetas = []
    for _ in range(args.muestras):
        serie, demanda = serie_sintetica(rng)
        stock = int(rng.integers(0, 80))
        minimo = int(rng.integers(0, 30))
        feats = features_de(serie, stock, minimo)
        filas.append([feats[n] for n in FEATURES])
        etiquetas.append(demanda)

    X = np.array(filas, dtype=float)
    y = np.array(etiquetas, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.semilla
    )

    modelo = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        max_depth=12,
        random_state=args.semilla,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": modelo,
            "features": FEATURES,
            "ventana": VENTANA,
            "mae_test": float(mae),
            "n_entrenamiento": int(len(X_train)),
            "tipo": "RandomForestRegressor",
            "objetivo": "demanda30",
        },
        salida,
    )
    print(f"Guardado {salida}")
    print(f"MAE test: {mae:.3f}  |  muestras train: {len(X_train)}")


if __name__ == "__main__":
    main()
