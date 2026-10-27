"""Proyección de demanda de inventario con Random Forest (regresión).

Si MODELO_DEMANDA_PATH apunta a un joblib válido, usa el modelo.
Si no, cae al stub (media de la serie × 30).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

log = logging.getLogger("ms-ia.demanda")

VENTANA = 90
FEATURES = [
    "suma7",
    "suma30",
    "media",
    "desvio",
    "maximo",
    "ultimo",
    "tendencia",
    "diasConSalida",
    "stockActual",
    "stockMinimo",
]


def serie_a_valores(serie: Any) -> list[float]:
    if not isinstance(serie, list):
        return []
    return [float(v) for v in serie if isinstance(v, (int, float))]


def normalizar_serie(valores: list[float], n: int = VENTANA) -> np.ndarray:
    arr = np.zeros(n, dtype=float)
    if not valores:
        return arr
    toma = valores[-n:]
    arr[-len(toma) :] = toma
    return arr


def features_de(valores: list[float], stock: int, minimo: int) -> dict[str, float]:
    serie = normalizar_serie(valores)
    cola7 = serie[-7:]
    cola30 = serie[-30:]
    mitad = len(serie) // 2
    media_ini = float(serie[:mitad].mean()) if mitad else 0.0
    media_fin = float(serie[mitad:].mean()) if mitad else 0.0
    return {
        "suma7": float(cola7.sum()),
        "suma30": float(cola30.sum()),
        "media": float(serie.mean()),
        "desvio": float(serie.std()),
        "maximo": float(serie.max()),
        "ultimo": float(serie[-1]),
        "tendencia": media_fin - media_ini,
        "diasConSalida": float((serie > 0).sum()),
        "stockActual": float(stock),
        "stockMinimo": float(minimo),
    }


def vector_features(item: dict[str, Any]) -> np.ndarray:
    valores = serie_a_valores(item.get("serieDiaria"))
    stock = int(item.get("stockActual") or 0)
    minimo = int(item.get("stockMinimo") or 0)
    feats = features_de(valores, stock, minimo)
    return np.array([feats[nombre] for nombre in FEATURES], dtype=float)


def proyectar_desde_demanda(demanda30: int, stock: int, minimo: int) -> dict[str, Any]:
    demanda = max(0, int(round(demanda30)))
    cobertura = round((stock / demanda) * 30) if demanda > 0 else None
    sugerido = max(0, minimo + demanda - stock)
    return {
        "demanda30": demanda,
        "coberturaDias": cobertura,
        "sugerido": sugerido,
    }


def stub_item(item: dict[str, Any]) -> dict[str, Any]:
    codigo = str(item.get("codigo") or "")
    valores = serie_a_valores(item.get("serieDiaria"))
    stock = int(item.get("stockActual") or 0)
    minimo = int(item.get("stockMinimo") or 0)
    if valores:
        demanda30 = round((sum(valores) / len(valores)) * 30)
    else:
        demanda30 = 0
    return {
        "codigo": codigo,
        **proyectar_desde_demanda(demanda30, stock, minimo),
        "fuente": "stub",
    }


class MotorDemanda:
    def __init__(self, modelo_path: str | None = None):
        self.modelo = None
        self.features = list(FEATURES)
        path = modelo_path or os.environ.get(
            "MODELO_DEMANDA_PATH",
            "/app/modelo/modelo_demanda.joblib",
        )
        archivo = Path(path)
        if not archivo.is_file():
            log.warning("Sin modelo de demanda en %s; se usa stub.", path)
            return
        info = joblib.load(archivo)
        self.modelo = info["modelo"]
        self.features = list(info.get("features", FEATURES))
        log.info("Modelo de demanda cargado desde %s", path)

    def predecir(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.modelo is None:
            return [stub_item(item) for item in items]

        proyecciones = []
        for item in items:
            codigo = str(item.get("codigo") or "")
            stock = int(item.get("stockActual") or 0)
            minimo = int(item.get("stockMinimo") or 0)
            vector = vector_features(item).reshape(1, -1)
            # Si el joblib guardó otro orden de features, rearmar.
            if self.features != FEATURES:
                feats = features_de(serie_a_valores(item.get("serieDiaria")), stock, minimo)
                vector = np.array([[feats[n] for n in self.features]], dtype=float)
            demanda = float(self.modelo.predict(vector)[0])
            proyecciones.append(
                {
                    "codigo": codigo,
                    **proyectar_desde_demanda(demanda, stock, minimo),
                    "fuente": "modelo",
                }
            )
        return proyecciones


_motor: MotorDemanda | None = None


def cargar_motor(modelo_path: str | None = None) -> MotorDemanda:
    global _motor
    _motor = MotorDemanda(modelo_path)
    return _motor


def predecir(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    motor = _motor or cargar_motor()
    return motor.predecir(items)
