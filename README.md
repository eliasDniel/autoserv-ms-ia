# autoserv-ms-ia

Diagnóstico del taller y proyección de demanda de inventario. Responde por NATS.

## Asuntos

- `ia.health` — confirma que el modelo de diagnóstico está cargado.
- `ia.diagnostico` — recibe `{ "texto": "..." }` y devuelve categoría, confianza, prioridad, síntomas, top 3, servicios y explicación.
- `ia.demanda.predecir` — recibe series diarias de salidas por repuesto y proyecta demanda a 30 días.

Climatización se resuelve por regla. Si la confianza queda bajo 0.30, el estado es no concluyente.

## Modelos (no van en Git)

Antes de construir la imagen hay que tener en `modelo/`:

- `modelo_diagnostico_rf.joblib` — Random Forest de fallas.
- `modelo_demanda.joblib` — Random Forest de demanda de repuestos.

Ambos están en `.gitignore` (`modelo/*.joblib`).

### Entrenar la demanda

Desde `ms-ia`, con las mismas dependencias del servicio:

```bash
python scripts/entrenar_demanda.py
```

Genera `modelo/modelo_demanda.joblib` con series sintéticas (Poisson) y un `RandomForestRegressor`.

### Variables

- `NATS_SERVICE`
- `MODELO_PATH` — diagnóstico
- `MODELO_DEMANDA_PATH` — demanda (si falta el archivo, se usa stub)
- `CATALOGO_PATH`

## Contrato demanda

Request:

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

Response:

```json
{
  "proyecciones": [
    {
      "codigo": "REP-001",
      "demanda30": 12,
      "coberturaDias": 105,
      "sugerido": 0,
      "fuente": "modelo"
    }
  ]
}
```

`fuente` puede ser `modelo` (joblib cargado) o `stub` (media de la serie × 30). El gateway, si ms-ia no responde, usa la proyección local de ms-core (`fuente: "local"`).
