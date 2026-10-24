import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PRIORIDAD = {
    "Frenos": "ALTA",
    "Neumáticos": "ALTA",
    "Suspensión y dirección": "ALTA",
    "Motor": "MEDIA",
    "Transmisión": "MEDIA",
    "Sistema eléctrico": "MEDIA",
    "Climatización": "BAJA",
}


class MotorDiagnostico:
    def __init__(self, modelo_path: str, catalogo_path: str):
        info = joblib.load(modelo_path)
        self.modelo = info["modelo"]
        self.sintomas_modelo = list(info["sintomas"])
        self.umbral = float(info["umbral"])
        self.metricas = {
            "nEntrenamiento": info["n_entrenamiento"],
            "f1MacroTest": info["f1_macro_test"],
            "top3Test": info["top3_test"],
            "f1MacroCv": info["f1_macro_cv"],
        }
        catalogo = json.loads(Path(catalogo_path).read_text(encoding="utf-8"))
        self.patrones = [
            (item["nombre"], re.compile(item["patron_es"]), bool(item["regla_climatizacion"]))
            for item in catalogo
        ]
        self.indice_modelo = {nombre: i for i, nombre in enumerate(self.sintomas_modelo)}
        self.clima_en_modelo = "Climatización" in list(self.modelo.classes_)

    def salud(self):
        return {
            "ok": True,
            "categorias": list(self.modelo.classes_),
            "variables": len(self.sintomas_modelo),
            "umbral": self.umbral,
            **self.metricas,
        }

    def diagnosticar(self, texto: str):
        fragmentos = dividir_problemas(texto)
        if not fragmentos:
            return [sin_sintomas(texto)]
        return [fila for fragmento in fragmentos for fila in self.clasificar(fragmento)]

    def clasificar(self, fragmento: str):
        detectados = self.extraer(fragmento)
        resultados = []
        clima = [nombre for nombre, es_clima in detectados if es_clima]
        if clima and not self.clima_en_modelo:
            resultados.append(
                {
                    "fragmento": fragmento,
                    "estado": "Diagnóstico por regla",
                    "sintomas": clima,
                    "categoria": "Climatización",
                    "confianza": None,
                    "prioridad": PRIORIDAD["Climatización"],
                    "top": [],
                }
            )

        del_modelo = [nombre for nombre, es_clima in detectados if not es_clima]
        if del_modelo:
            vector = np.zeros((1, len(self.sintomas_modelo)))
            vector[0, [self.indice_modelo[nombre] for nombre in del_modelo]] = 1
            tabla = pd.DataFrame(vector, columns=self.sintomas_modelo)
            proba = self.modelo.predict_proba(tabla)[0]
            orden = np.argsort(proba)[::-1][:3]
            top = [
                {
                    "categoria": str(self.modelo.classes_[i]),
                    "confianza": round(float(proba[i]), 3),
                }
                for i in orden
            ]
            categoria = top[0]["categoria"]
            confianza = top[0]["confianza"]
            concluyente = confianza >= self.umbral
            resultados.append(
                {
                    "fragmento": fragmento,
                    "estado": (
                        "Diagnóstico sugerido"
                        if concluyente
                        else "No concluyente: derivar a revisión mecánica"
                    ),
                    "sintomas": del_modelo,
                    "categoria": categoria,
                    "confianza": confianza,
                    "prioridad": PRIORIDAD.get(categoria) if concluyente else None,
                    "top": top,
                }
            )

        return resultados or [sin_sintomas(fragmento)]

    def extraer(self, texto: str):
        normalizado = normalizar(texto)
        return [
            (nombre, es_clima)
            for nombre, patron, es_clima in self.patrones
            if patron.search(normalizado)
        ]


def normalizar(texto: str) -> str:
    sin_tilde = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sin_tilde).strip().lower()


def dividir_problemas(texto: str):
    partes = re.split(r"[\n.;]+", str(texto))
    return [parte.strip() for parte in partes if len(parte.strip()) >= 3]


def sin_sintomas(fragmento: str):
    return {
        "fragmento": fragmento,
        "estado": "Sin síntomas de falla reconocidos",
        "sintomas": [],
        "categoria": None,
        "confianza": None,
        "prioridad": None,
        "top": [],
    }
