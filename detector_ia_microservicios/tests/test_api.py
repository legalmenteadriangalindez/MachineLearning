import numpy as np
from fastapi.testclient import TestClient

import backend.main as modulo_api

cliente = TestClient(modulo_api.app)
TEXTO = "Este es un texto lo suficientemente extenso para probar la solicitud de análisis de la API."


class ModeloFalso:
    classes_ = np.array([0, 1])

    def predict_proba(self, textos):
        return np.array([[0.25, 0.75]])


def test_health():
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "ok"


def test_sin_modelo(monkeypatch):
    monkeypatch.setattr(modulo_api, "obtener_modelo", lambda: None)
    respuesta = cliente.post("/analizar", json={"texto": TEXTO})
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["estado"] == "sin_modelo"
    assert datos["etiqueta"] is None
    assert datos["puntaje_ia"] is None
    assert datos["estadisticas"]["palabras"] > 0


def test_modelo_simulado(monkeypatch):
    monkeypatch.setattr(modulo_api, "obtener_modelo", lambda: ModeloFalso())
    respuesta = cliente.post("/analizar", json={"texto": TEXTO})
    assert respuesta.status_code == 200
    assert respuesta.json()["etiqueta"] == "IA"
    assert respuesta.json()["puntaje_ia"] == 0.75


def test_texto_corto_rechazado():
    respuesta = cliente.post("/analizar", json={"texto": "hola"})
    assert respuesta.status_code == 422


def test_texto_blanco_rechazado():
    respuesta = cliente.post("/analizar", json={"texto": " " * 100})
    assert respuesta.status_code == 422
