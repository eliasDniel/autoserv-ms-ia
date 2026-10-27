# autoserv-ms-ia

Diagnóstico del taller y proyección de demanda de inventario. Responde por NATS.

## Asuntos

- `ia.health` — confirma que el modelo de diagnóstico está cargado.
- `ia.diagnostico` — recibe `{ "texto": "..." }` y devuelve categoría, confianza, prioridad, síntomas, top 3, servicios sugeridos y una explicación armada con ese resultado.
- `ia.demanda.predecir` — recibe series diarias de salidas por repuesto y devuelve proyección a 30 días.

Climatización se resuelve por regla. Si la confianza queda bajo 0.30, el estado es no concluyente.

## Diagnóstico

El archivo `modelo/modelo_diagnostico_rf.joblib` no va en Git. Hay que copiarlo a esa carpeta antes de construir la imagen.

Variables:

- `MODELO_PATH` — ruta al joblib de diagnóstico (default `/app/modelo/modelo_diagnostico_rf.joblib`)
- `CATALOGO_PATH` — catálogo de síntomas
- `NATS_SERVICE` — URL de NATS

## Demanda (pendiente de modelo)

Hoy `app/demanda.py` es un **stub**: media de la serie diaria × 30, con `fuente: "stub"`.

Contrato request:

```json
{
  "items": [
    {
      "codigo": "REP-001",
      "serieDiaria": [0, 1, 0, 2],
      "stockActual": 42,
      "stockMinimo": 20
    }
  ]
}
```

Contrato response:

```json
{
  "proyecciones": [
    {
      "codigo": "REP-001",
      "demanda30": 12,
      "coberturaDias": 105,
      "sugerido": 0,
      "fuente": "stub"
    }
  ]
}
```

Para enchufar el modelo más adelante:

1. Entrenar un regressor sklearn y guardar `modelo/modelo_demanda.joblib` (gitignore igual que el de diagnóstico).
2. Definir `MODELO_DEMANDA_PATH` en el entorno / compose.
3. En `app/demanda.py`, cargar el joblib y devolver `fuente: "modelo"` manteniendo las mismas claves del contrato.
4. El gateway ya llama a `ia.demanda.predecir` al listar inventario; si ms-ia no responde, usa la proyección local de ms-core (`fuente: "local"`).
