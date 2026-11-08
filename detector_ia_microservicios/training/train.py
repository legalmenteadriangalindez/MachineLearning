"""Entrenamiento reproducible con CSV etiquetado de procedencia comprobada."""
import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]


def leer_y_validar(ruta: Path) -> pd.DataFrame:
    datos = pd.read_csv(ruta)
    if not {"texto", "etiqueta"}.issubset(datos.columns):
        raise ValueError("El CSV debe incluir las columnas 'texto' y 'etiqueta'.")
    datos = datos[["texto", "etiqueta"]].copy()
    if datos.isna().any().any():
        raise ValueError("Hay textos o etiquetas vacíos. Corrígelos antes de entrenar.")
    if not datos["texto"].map(lambda valor: isinstance(valor, str)).all():
        raise ValueError("Todos los textos deben ser cadenas de caracteres.")
    datos["texto"] = datos["texto"].str.strip()
    if (datos["texto"].str.len() < 50).any():
        raise ValueError("Todos los ejemplos deben tener al menos 50 caracteres.")
    etiquetas = pd.to_numeric(datos["etiqueta"], errors="coerce")
    if etiquetas.isna().any() or not etiquetas.isin([0, 1]).all():
        raise ValueError("Las etiquetas solo pueden ser 0 (humano) o 1 (IA).")
    datos["etiqueta"] = etiquetas.astype(int)

    # Quitar duplicados ANTES de separar entrenamiento y prueba.
    datos["normalizado"] = datos["texto"].str.replace(r"\s+", " ", regex=True).str.casefold()
    if (datos.groupby("normalizado")["etiqueta"].nunique() > 1).any():
        raise ValueError("Hay textos repetidos con etiquetas contradictorias.")
    datos = datos.drop_duplicates(subset="normalizado").drop(columns="normalizado")
    recuento = datos["etiqueta"].value_counts()
    if len(recuento) != 2 or (recuento < 10).any():
        raise ValueError("Se necesitan al menos 10 textos distintos por clase para esta prueba técnica.")
    return datos


def entrenar(ruta_csv: Path, ruta_salida: Path):
    datos = leer_y_validar(ruta_csv)
    x_ent, x_prueba, y_ent, y_prueba = train_test_split(
        datos["texto"], datos["etiqueta"], test_size=0.2,
        stratify=datos["etiqueta"], random_state=42,
    )
    modelo = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40_000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    modelo.fit(x_ent, y_ent)
    predicciones = modelo.predict(x_prueba)
    informe = {
        "ejemplos_total": len(datos),
        "ejemplos_entrenamiento": len(x_ent),
        "ejemplos_prueba": len(x_prueba),
        "accuracy": float(accuracy_score(y_prueba, predicciones)),
        "classification_report": classification_report(
            y_prueba, predicciones, labels=[0, 1],
            target_names=["Humano", "IA"], output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_prueba, predicciones, labels=[0, 1]).tolist(),
        "aviso": "Evaluación interna; puede fallar ante autores, temas o modelos diferentes.",
    }

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta_salida.with_name(ruta_salida.name + ".tmp")
    joblib.dump(modelo, temporal)
    os.replace(temporal, ruta_salida)
    ruta_metricas = ruta_salida.with_name("metricas.json")
    ruta_metricas.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo guardado en: {ruta_salida}")
    print(f"Métricas guardadas en: {ruta_metricas}")
    print(f"Exactitud interna: {informe['accuracy']:.3f}")
    return informe


def main():
    parser = argparse.ArgumentParser(description="Entrenar clasificador experimental IA vs Humano")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "dataset.csv")
    parser.add_argument("--modelo", type=Path, default=Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "detector.joblib"))))
    args = parser.parse_args()
    entrenar(args.csv, args.modelo)


if __name__ == "__main__":
    main()
