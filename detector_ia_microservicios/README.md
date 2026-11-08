# Detector experimental de texto IA — FastAPI + Streamlit

Proyecto gratuito, educativo y modular con dos servicios independientes:

- **API (FastAPI):** recibe textos, devuelve estadísticas y consulta un clasificador local entrenable.
- **Interfaz (Streamlit):** envía solicitudes HTTP a la API y presenta resultados.
- **Entrenamiento (Scikit-learn):** TF-IDF + regresión logística.
- **Pruebas (Pytest):** validan API y funcionamiento del entrenamiento.

> **Advertencia:** los detectores de IA se equivocan. La clasificación NO acredita autoría ni debe utilizarse para sancionar a personas. El puntaje no es una probabilidad verificada de autoría.

## Requisitos
- Python 3.11 recomendado; Windows, Linux o macOS.
- Docker Compose opcional.
- Sin claves API ni GPU.

## Iniciar en Windows (PowerShell)

Desde esta carpeta:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Si no tienes py -3.11, utiliza python -m venv .venv.

En una terminal con el entorno activado:

~~~powershell
uvicorn backend.main:app --reload --port 8000
~~~

En otra terminal con el entorno activado:

~~~powershell
streamlit run frontend/app.py --server.port 8501
~~~

- **Interfaz:** http://localhost:8501
- **Documentación API:** http://localhost:8000/docs
- **Salud de API:** http://localhost:8000/health
- **Modelo:** http://localhost:8000/modelo

Sin modelo entrenado, la API solo muestra métricas descriptivas.

## Iniciar con Docker Compose

~~~powershell
docker compose up --build -d
~~~

Para revisar registros o detener:

~~~powershell
docker compose logs -f api
docker compose logs -f interfaz
docker compose down
~~~

## Entrenar el detector

Prepara data/dataset.csv en UTF-8 con columnas **texto** y **etiqueta**:
- 0 = texto humano con origen verificado.
- 1 = texto generado por IA con origen verificado.

Consulta data/FORMATO_CSV.md y ejecuta:

~~~powershell
python -m training.train --csv data/dataset.csv
~~~

Se crearán models/detector.joblib y models/metricas.json. La API puede recargar el modelo al modificarse el archivo.

Para entrenar exclusivamente con Docker:

~~~powershell
docker compose run --rm --no-deps api python -m training.train --csv data/dataset.csv
~~~

**Importante:** el mínimo de 10 textos por clase solo sirve para verificar funcionamiento, NO para validar un detector. Para investigación necesitas muestras diversas y una evaluación externa independiente, incluyendo falsos positivos, F1 y sesgos por idioma y dominio.

## Probar la API

Desde http://localhost:8000/docs ejecuta POST /analizar con:

~~~json
{"texto":"Este texto de ejemplo tiene suficiente longitud para comprobar que la API recibe y procesa correctamente una solicitud."}
~~~

Sin modelo, devuelve estado sin_modelo y estadísticas; con un modelo entrenado, devuelve estado analizado, etiqueta y puntaje_ia.

## Pruebas automáticas

~~~powershell
python -m pytest -q
~~~

Las pruebas incluyen modelos y textos ficticios exclusivamente para comprobar el software, no la eficacia de detección.

## Endpoints

| Método | Ruta | Función |
| --- | --- | --- |
| GET | /health | Salud del microservicio |
| GET | /modelo | Disponibilidad del modelo |
| POST | /analizar | Analizar un texto |

## Seguridad y próximos pasos

- No publicar directamente la API en internet sin autenticación, límites de uso y monitoreo.
- Esta versión no guarda el texto de las solicitudes en base de datos.
- No cargar archivos .joblib desconocidos (joblib utiliza pickle).
- El modelo no debe decidir autoría humana.
- Futuras mejoras: dataset fiable, evaluación externa, modelos Transformer, carga de PDF/DOCX y despliegue controlado.
