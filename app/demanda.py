"""Proyección de demanda de inventario.

Hoy es un stub: media de la serie diaria × 30.
Cuando exista el modelo, cargar el joblib aquí (MODELO_DEMANDA_PATH)
y sustituir el cuerpo de predecir() manteniendo el mismo contrato.
"""

from __future__ import annotations

from typing import Any


def predecir(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Punto de enganche futuro (MODELO_DEMANDA_PATH + joblib.load):
    # return [{..., "fuente": "modelo"}, ...]

    proyecciones = []
    for item in items:
        codigo = str(item.get("codigo") or "")
        serie = item.get("serieDiaria") or []
        if not isinstance(serie, list):
            serie = []
        valores = [float(v) for v in serie if isinstance(v, (int, float))]
        stock = int(item.get("stockActual") or 0)
        minimo = int(item.get("stockMinimo") or 0)

        if valores:
            media_diaria = sum(valores) / len(valores)
            demanda30 = round(media_diaria * 30)
        else:
            demanda30 = 0

        cobertura = round((stock / demanda30) * 30) if demanda30 > 0 else None
        sugerido = max(0, minimo + demanda30 - stock)

        proyecciones.append(
            {
                "codigo": codigo,
                "demanda30": demanda30,
                "coberturaDias": cobertura,
                "sugerido": sugerido,
                "fuente": "stub",
            }
        )
    return proyecciones
