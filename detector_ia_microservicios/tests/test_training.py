import pandas as pd
import pytest

from training.train import entrenar, leer_y_validar


def crear_datos():
    # Textos ficticios SOLO para verificar que el software entrena.
    # NO representan autoría real ni permiten medir calidad de un detector.
    humanos = [
        {"texto": f"Mi historia personal numero {n} relata una salida al parque con mi familia y una tarde de lluvia intensa.", "etiqueta": 0}
        for n in range(20)
    ]
    ia = [
        {"texto": f"Documento automatizado numero {n} analiza una secuencia ordenada de conceptos y presenta una conclusión general.", "etiqueta": 1}
        for n in range(20)
    ]
    return pd.DataFrame(humanos + ia)


def test_entrenamiento_tecnico(tmp_path):
    ruta_csv = tmp_path / "dataset.csv"
    crear_datos().to_csv(ruta_csv, index=False)
    ruta_modelo = tmp_path / "detector.joblib"
    informe = entrenar(ruta_csv, ruta_modelo)
    assert informe["ejemplos_total"] == 40
    assert ruta_modelo.exists()
    assert (tmp_path / "metricas.json").exists()


def test_etiqueta_contradictoria(tmp_path):
    datos = crear_datos()
    datos.loc[1, "texto"] = datos.loc[0, "texto"]
    datos.loc[1, "etiqueta"] = 1
    ruta_csv = tmp_path / "dataset.csv"
    datos.to_csv(ruta_csv, index=False)
    with pytest.raises(ValueError, match="contradictorias"):
        leer_y_validar(ruta_csv)
