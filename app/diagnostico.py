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

ETIQUETA = {"ALTA": "Alta", "MEDIA": "Media", "BAJA": "Baja"}

SERVICIOS = {
    "Frenos": [
        {"code": "SRV-005", "name": "Cambio de pastillas de freno", "time": "1.5h", "price": 150},
        {"code": "SRV-011", "name": "Revisión del sistema de frenos", "time": "1.5h", "price": 45},
        {"code": "SRV-008", "name": "Diagnóstico computarizado", "time": "0.5h", "price": 35},
    ],
    "Motor": [
        {"code": "SRV-001", "name": "Cambio de aceite y filtro", "time": "1h", "price": 45},
        {"code": "SRV-006", "name": "Revisión general del vehículo", "time": "1h", "price": 80},
        {"code": "SRV-008", "name": "Diagnóstico computarizado", "time": "0.5h", "price": 35},
    ],
    "Neumáticos": [
        {"code": "SRV-004", "name": "Revisión y calibración de llantas", "time": "0.5h", "price": 20},
        {"code": "SRV-007", "name": "Alineación y balanceo", "time": "1h", "price": 40},
    ],
    "Sistema eléctrico": [
        {"code": "SRV-012", "name": "Revisión del sistema eléctrico", "time": "1.5h", "price": 55},
        {"code": "SRV-008", "name": "Diagnóstico computarizado", "time": "0.5h", "price": 35},
    ],
    "Suspensión y dirección": [
        {"code": "SRV-002", "name": "Revisión de suspensión delantera", "time": "2h", "price": 80},
        {"code": "SRV-013", "name": "Revisión de dirección", "time": "1.5h", "price": 70},
        {"code": "SRV-007", "name": "Alineación y balanceo", "time": "1h", "price": 40},
    ],
    "Transmisión": [
        {"code": "SRV-014", "name": "Revisión de transmisión", "time": "2h", "price": 90},
        {"code": "SRV-008", "name": "Diagnóstico computarizado", "time": "0.5h", "price": 35},
    ],
    "Climatización": [
        {"code": "SRV-003", "name": "Revisión sistema A/C", "time": "1.5h", "price": 60},
        {"code": "SRV-015", "name": "Recarga de refrigerante", "time": "0.5h", "price": 45},
    ],
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
            return [completar(sin_sintomas(texto))]
        return [
            completar(fila)
            for fragmento in fragmentos
            for fila in self.clasificar(fragmento)
        ]

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


def completar(fila):
    confianza = fila["confianza"]
    fila["category"] = fila["categoria"]
    fila["confidence"] = None if confianza is None else round(confianza * 100)
    fila["description"] = ", ".join(fila["sintomas"]) if fila["sintomas"] else fila["estado"]
    fila["priority"] = ETIQUETA.get(fila["prioridad"])
    fila["services"] = [dict(item) for item in SERVICIOS.get(fila["categoria"], [])] if fila["categoria"] else []
    fila["explanation"] = explicar(fila)
    return fila


def explicar(fila):
    sintomas = ", ".join(fila["sintomas"]) if fila["sintomas"] else "ningún síntoma del catálogo"
    if fila["categoria"] is None:
        return (
            "El texto no coincide con síntomas del catálogo. "
            "Describe el ruido, la luz del tablero o la falla con más detalle."
        )
    if fila["confianza"] is None:
        return (
            f"Se detectó {sintomas}. Climatización no sale del Random Forest: "
            "ese sistema se asigna por regla. Conviene revisar el aire acondicionado "
            "antes de sustituir piezas. Es una orientación preliminar y no reemplaza al técnico."
        )
    porcentaje = round(fila["confianza"] * 100)
    alternativas = ", ".join(
        f"{item['categoria']} ({round(item['confianza'] * 100)}%)" for item in fila["top"][1:]
    )
    texto = (
        f"El modelo asigna {fila['categoria']} con {porcentaje}% de confianza "
        f"a partir de: {sintomas}."
    )
    if alternativas:
        texto += f" Otras posibilidades: {alternativas}."
    if fila["prioridad"]:
        texto += f" La prioridad de esta categoría es {fila['prioridad'].lower()}."
    if "No concluyente" in fila["estado"]:
        texto += (
            " La confianza está bajo el umbral de 30%, así que conviene una revisión "
            "mecánica antes de decidir el trabajo."
        )
    texto += " Es una orientación preliminar y no reemplaza al técnico."
    return texto
