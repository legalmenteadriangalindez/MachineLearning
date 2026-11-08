"""Carga un modelo entrenado LOCALMENTE. No cargar archivos .joblib de terceros."""
import os
from functools import lru_cache
from pathlib import Path

import joblib

RUTA_POR_DEFECTO = Path(__file__).resolve().parents[1] / "models" / "detector.joblib"


def ruta_modelo() -> Path:
    return Path(os.getenv("MODEL_PATH", str(RUTA_POR_DEFECTO))).resolve()


@lru_cache(maxsize=4)
def _cargar_modelo(ruta: str, ultima_modificacion_ns: int):
    # joblib usa pickle: esta ruta debe contener SOLO modelos propios y confiables.
    return joblib.load(ruta)


def obtener_modelo():
    ruta = ruta_modelo()
    if not ruta.is_file():
        return None
    # Cambios en mtime permiten recargar el modelo luego de un entrenamiento.
    return _cargar_modelo(str(ruta), ruta.stat().st_mtime_ns)
