from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler  # <--- Agregado
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los datos desde el archivo CSV
# Cambia 'tus_datos.csv' por el nombre real de tu archivo
df = pd.read_csv('tus_datos.csv')

# Seleccionar variables predictoras y objetivo
X = df[['manzana', 'naranja']]   # Variables predictoras
y = df['clase']                  # Variable objetivo

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Escalado de características (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Ajusta y transforma entrenamiento
X_test_scaled = scaler.transform(X_test)        # Solo transforma prueba (evita data leakage)

# 4. Crear y entrenar el modelo KNN usando los datos escalados
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_train_scaled, y_train)

# 5. Predicciones usando los datos escalados
y_pred = modelo.predict(X_test_scaled)
y_prob = modelo.predict_proba(X_test_scaled)[:, 1]   # Probabilidad de la clase positiva

# 6. Cálculo de métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {auc:.2f}")