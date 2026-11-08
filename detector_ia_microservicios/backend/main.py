"""Microservicio REST: el frontend solo envía textos, no entrena modelos."""
import re

from fastapi import FastAPI, HTTPException

from backend.modelo import obtener_modelo
from backend.schemas import AnalisisSalida, Estadisticas, TextoEntrada

app = FastAPI(
    title="Detector educativo de texto IA",
    description="API de clasificación experimental; NO acredita autoría de textos.",
    version="0.1.0",
)


def calcular_estadisticas(texto: str) -> Estadisticas:
    palabras = re.findall(r"\b\w+\b", texto.casefold(), flags=re.UNICODE)
    diversidad = round(len(set(palabras)) / len(palabras), 3) if palabras else 0.0
    return Estadisticas(
        caracteres=len(texto),
        palabras=len(palabras),
        diversidad_lexica=diversidad,
    )


def modelo_seguro():
    try:
        return obtener_modelo()
    except Exception as error:
        # No se expone al cliente el sistema de archivos ni el detalle interno.
        raise HTTPException(
            status_code=503,
            detail="No fue posible cargar el modelo local. Revisa el entrenamiento y los registros del servidor.",
        ) from error


@app.get("/health")
def health():
    return {"servicio": "detector-ia-api", "estado": "ok"}


@app.get("/modelo")
def estado_modelo():
    disponible = modelo_seguro() is not None
    return {
        "modelo_entrenado": disponible,
        "mensaje": (
            "Modelo local cargado; resultados solo orientativos."
            if disponible
            else "No hay un modelo entrenado. La API puede mostrar estadísticas, pero no clasificar."
        ),
    }


@app.post("/analizar", response_model=AnalisisSalida)
def analizar(entrada: TextoEntrada):
    estadisticas = calcular_estadisticas(entrada.texto)
    modelo = modelo_seguro()
    if modelo is None:
        return AnalisisSalida(
            estado="sin_modelo",
            estadisticas=estadisticas,
            mensaje="Aún no hay un modelo entrenado. Ejecuta el comando de entrenamiento con un CSV etiquetado.",
        )

    try:
        clases = [int(clase) for clase in modelo.classes_]
        indice_ia = clases.index(1)
        puntaje = float(modelo.predict_proba([entrada.texto])[0][indice_ia])
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="El modelo no tiene el formato esperado para clasificar textos.",
        ) from error

    return AnalisisSalida(
        estado="analizado",
        estadisticas=estadisticas,
        etiqueta="IA" if puntaje >= 0.5 else "Humano",
        puntaje_ia=round(puntaje, 4),
        mensaje=(
            "Resultado experimental: la puntuación pertenece al clasificador y NO "
            "es una prueba ni una probabilidad verificada de autoría."
        ),
    )
