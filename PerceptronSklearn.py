import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================================================
# 1. Carga de Datos (Simulación de read_csv)
# =====================================================================
# Nota: Reemplaza 'dataset.csv' por la ruta real de tu archivo.
# Tu CSV debería tener las columnas de características y una para la etiqueta (target).

# Para que el código funcione directamente, creamos un archivo CSV de prueba rápido:
data_ejemplo = """feature1,feature2,target
0,0,0
0,1,1
1,0,1
1,1,1"""
with open("dataset.csv", "w") as f:
    f.write(data_ejemplo)

# --- Carga real usando pandas ---
df = pd.read_csv("dataset.csv")

# Separamos las características (X) de la variable objetivo (y)
X = df[["feature1", "feature2"]].values
y = df["target"].values

# Dividimos en entrenamiento y prueba para una evaluación real
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =====================================================================
# 2. Configuración del Modelo (Parámetros más importantes)
# =====================================================================
modelo = Perceptron(
    max_iter=1000,  # Épocas máximas si no converge antes
    eta0=0.1,  # Tasa de aprendizaje (controla la velocidad/estabilidad)
    tol=1e-3,  # Detiene el entrenamiento si la mejora es menor a esto
    shuffle=True,  # Mezcla los datos en cada época para evitar sesgos por orden
    fit_intercept=True,  # Permite aprender el Bias (la frontera no se fuerza al origen)
    random_state=42,  # Semilla para que los resultados sean reproducibles
)

# =====================================================================
# 3. Entrenamiento
# =====================================================================
modelo.fit(X_train, y_train)

# =====================================================================
# 4. Predicciones
# =====================================================================
predicciones_train = modelo.predict(X_train)
predicciones_test = modelo.predict(X_test)

# =====================================================================
# 5. Parámetros Aprendidos por el Algoritmo
# =====================================================================
print("=== PARÁMETROS APRENDIDOS ===")
print(f"Pesos finales (w): {modelo.coef_}")
print(f"Sesgo final (bias / b): {modelo.intercept_}")
print("-" * 50)

# =====================================================================
# 6. Evaluación del Modelo (Métricas)
# =====================================================================
print("=== EVALUACIÓN DEL MODELO (Datos de Prueba) ===")

# Accuracy
exactitud = accuracy_score(y_test, predicciones_test)
print(f"Accuracy (Exactitud): {exactitud:.2f} ({exactitud * 100}%)\n")

# Matriz de Confusión
print("Matriz de Confusión:")
print(confusion_matrix(y_test, predicciones_test))
print("\n" + "-" * 50)

# Reporte de Clasificación Completo (Precision, Recall, F1-Score)
print("Reporte de Clasificación:")
print(classification_report(y_test, predicciones_test, zero_division=0))