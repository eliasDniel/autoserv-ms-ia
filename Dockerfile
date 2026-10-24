FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY datos ./datos
COPY modelo ./modelo

ENV NATS_SERVICE=nats://nats:4222
ENV MODELO_PATH=/app/modelo/modelo_diagnostico_rf.joblib
ENV CATALOGO_PATH=/app/datos/catalogo_sintomas.json

CMD ["python", "-m", "app.main"]
