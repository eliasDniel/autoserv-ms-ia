# autoserv-ms-ia

Diagnóstico del taller. Carga el Random Forest ya entrenado y responde por NATS.

- `ia.health` — confirma que el modelo está cargado.
- `ia.diagnostico` — recibe `{ "texto": "..." }` y devuelve categoría, confianza, prioridad, síntomas, top 3, servicios sugeridos y una explicación armada con ese resultado.

Climatización se resuelve por regla. Si la confianza queda bajo 0.30, el estado es no concluyente.

El archivo `modelo/modelo_diagnostico_rf.joblib` no va en Git. Hay que copiarlo a esa carpeta antes de construir la imagen.
