"""Contratos públicos de la API."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TextoEntrada(BaseModel):
    texto: str = Field(..., min_length=50, max_length=20_000)

    @field_validator("texto")
    @classmethod
    def validar_texto(cls, valor: str) -> str:
        valor = valor.strip()
        if len(valor) < 50:
            raise ValueError("El texto debe tener al menos 50 caracteres útiles.")
        return valor


class Estadisticas(BaseModel):
    caracteres: int
    palabras: int
    diversidad_lexica: float


class AnalisisSalida(BaseModel):
    estado: Literal["sin_modelo", "analizado"]
    estadisticas: Estadisticas
    etiqueta: Literal["IA", "Humano"] | None = None
    puntaje_ia: float | None = None
    mensaje: str
