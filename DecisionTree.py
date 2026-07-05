import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler  # <--- Agregado
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Cargar los datos desde el archivo CSV
# Cambia 'tus_datos.csv' por el nombre real de tu archivo
df = pd.read_csv('tus_datos.csv')

# Seleccionar variables predictoras y objetivo (ajusta los nombres de las columnas a tu CSV)
X = df[['manzana', 'naranja']]   # Variables predictoras
y = df['clase']                  # Variable objetivo

# 2. Dividir en entrenamiento y prueba (test_size=0.3 según tu código)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Escalado de características (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajusta y transforma entrenamiento
X_test_scaled = scaler.transform(X_test)        # Solo transforma prueba

# 4. Crear y entrenar el modelo de Árbol de Decisión usando los datos escalados
modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_train_scaled, y_train)

# 5. Predecir usando los datos escalados
y_pred = modelo.predict(X_test_scaled)
y_prob = modelo.predict_proba(X_test_scaled)[:, 1]   # Probabilidad de clase positiva

# 6. Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Matriz de Confusión - Árbol de Decisión (Datos Escalados)')
plt.show()

# 7. Métricas adicionales
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")