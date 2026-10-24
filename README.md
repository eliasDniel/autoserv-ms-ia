# autoserv-ms-ia

Diagnóstico del taller. Carga el Random Forest ya entrenado y responde por NATS.

- `ia.health` — confirma que el modelo está cargado.
- `ia.diagnostico` — recibe `{ "texto": "..." }` y devuelve categoría, confianza, prioridad, síntomas detectados y el top 3.

Climatización se resuelve por regla. Si la confianza queda bajo 0.30, el estado es no concluyente.
