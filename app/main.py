import asyncio
import json
import logging
import os
import signal

from nats.aio.client import Client as NATS

from app.demanda import predecir as predecir_demanda
from app.diagnostico import MotorDiagnostico

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ms-ia")

ASUNTOS = ("ia.health", "ia.diagnostico", "ia.demanda.predecir")


def responder(packet, msg, resultado=None, error=None):
    cuerpo = {"id": packet.get("id"), "isDisposed": True}
    if error:
        cuerpo["err"] = {"status": "error", "message": error}
    else:
        cuerpo["response"] = resultado
    return json.dumps(cuerpo, ensure_ascii=False).encode()


async def main():
    motor = MotorDiagnostico(
        os.environ.get("MODELO_PATH", "/app/modelo/modelo_diagnostico_rf.joblib"),
        os.environ.get("CATALOGO_PATH", "/app/datos/catalogo_sintomas.json"),
    )
    log.info(
        "Modelo cargado: %s variables, umbral %s",
        len(motor.sintomas_modelo),
        motor.umbral,
    )

    nc = NATS()
    url = os.environ.get("NATS_SERVICE", "nats://nats:4222")
    await nc.connect(url, name="autoserv-ms-ia")
    log.info("Conectado a %s", url)

    async def atender(msg):
        try:
            packet = json.loads(msg.data.decode())
        except json.JSONDecodeError:
            packet = {}
        if not isinstance(packet, dict):
            packet = {}
        patron = packet.get("pattern") or msg.subject
        datos = packet.get("data") if isinstance(packet.get("data"), dict) else {}
        try:
            if patron == "ia.health":
                salida = responder(packet, msg, motor.salud())
            elif patron == "ia.diagnostico":
                texto = datos.get("texto")
                if not isinstance(texto, str) or not texto.strip():
                    raise ValueError("Escribe el síntoma del vehículo.")
                salida = responder(packet, msg, {"resultados": motor.diagnosticar(texto)})
            elif patron == "ia.demanda.predecir":
                items = datos.get("items")
                if not isinstance(items, list):
                    raise ValueError("Envía la lista de ítems con serie diaria.")
                salida = responder(
                    packet,
                    msg,
                    {"proyecciones": predecir_demanda(items)},
                )
            else:
                raise ValueError(f"Patrón no reconocido: {patron}")
        except ValueError as exc:
            salida = responder(packet, msg, error=str(exc))
        except Exception:
            log.exception("Fallo en ms-ia")
            salida = responder(packet, msg, error="No se pudo procesar la solicitud de IA.")
        if msg.reply:
            await msg.respond(salida)

    for asunto in ASUNTOS:
        await nc.subscribe(asunto, cb=atender)
        log.info("Escuchando %s", asunto)

    detener = asyncio.Event()
    bucle = asyncio.get_running_loop()
    for senal in (signal.SIGINT, signal.SIGTERM):
        bucle.add_signal_handler(senal, detener.set)
    await detener.wait()
    await nc.drain()


if __name__ == "__main__":
    asyncio.run(main())
