"""Cliente Streamlit de la API FastAPI."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Detector IA | Base", page_icon="🔎", layout="centered")
st.title("🔎 Detector experimental de textos con IA")
st.caption("Python + FastAPI + Streamlit | Proyecto educativo, sin servicios de pago")
st.info(
    "Un detector de IA no puede demostrar la autoría de un texto. "
    "No utilices sus resultados como prueba para sancionar o acusar a una persona."
)

try:
    respuesta_estado = requests.get(f"{API_URL}/modelo", timeout=3)
    respuesta_estado.raise_for_status()
    estado = respuesta_estado.json()
    if estado.get("modelo_entrenado"):
        st.success("Microservicio disponible · Modelo cargado")
    else:
        st.warning("Microservicio disponible · Modelo aún no entrenado")
except requests.RequestException:
    st.error("El microservicio no responde. Inicia FastAPI en el puerto 8000.")

archivo = st.file_uploader("Opcional: carga un archivo de texto (.txt)", type=["txt"])
texto_archivo = ""
if archivo is not None:
    try:
        texto_archivo = archivo.getvalue().decode("utf-8-sig")
        if len(texto_archivo) > 20_000:
            st.warning("El archivo excede 20.000 caracteres. Se tomará solo el inicio.")
            texto_archivo = texto_archivo[:20_000]
    except UnicodeDecodeError:
        st.error("El archivo debe estar codificado en UTF-8.")

with st.form("formulario_analisis"):
    texto = st.text_area(
        "Texto que deseas analizar",
        value=texto_archivo,
        height=230,
        placeholder="Pega aquí un texto de al menos 50 caracteres...",
        max_chars=20_000,
    )
    enviar = st.form_submit_button("Analizar texto", use_container_width=True)

if enviar:
    if len(texto.strip()) < 50:
        st.warning("Escribe al menos 50 caracteres útiles.")
    else:
        try:
            respuesta = requests.post(
                f"{API_URL}/analizar",
                json={"texto": texto},
                timeout=15,
            )
            respuesta.raise_for_status()
            datos = respuesta.json()
            st.subheader("Resultado")
            estadisticas = datos["estadisticas"]
            a, b, c = st.columns(3)
            a.metric("Palabras", estadisticas["palabras"])
            b.metric("Caracteres", estadisticas["caracteres"])
            c.metric("Diversidad léxica", estadisticas["diversidad_lexica"])

            if datos["estado"] == "sin_modelo":
                st.info(datos["mensaje"])
                st.caption("Estas estadísticas describen el texto; no detectan su autoría.")
            else:
                st.write(f"**Clasificación orientativa:** {datos['etiqueta']}")
                puntaje_ia = datos["puntaje_ia"]
                st.metric("Puntaje interno del modelo hacia la clase IA", f"{puntaje_ia:.1%}")
                st.progress(puntaje_ia)
                st.warning(datos["mensaje"])
        except requests.HTTPError as error:
            st.error(f"La API devolvió un error: {error.response.status_code}")
        except requests.RequestException:
            st.error("Error de conexión o tiempo de espera agotado al consultar la API.")

with st.expander("¿Cómo se entrena el detector?"):
    st.code("python -m training.train --csv data/dataset.csv", language="bash")
    st.write("El CSV debe tener las columnas `texto` y `etiqueta` (0=humano, 1=IA).")
    st.write("Se requieren textos con origen comprobado; la clasificación es experimental.")
