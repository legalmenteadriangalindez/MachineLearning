import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Cargar el dataset desde un archivo CSV
# Reemplaza 'datos_estudiantes.csv' por el nombre real de tu archivo
df = pd.read_csv("datos_estudiantes.csv")

# 2. Definir variables
X = df.drop("aprueba", axis=1)  # Variables predictoras (todas menos la objetivo)
y = df["aprueba"]  # Variable objetivo

# 3. Dividir en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Escalado de características (Paso crítico para que SVM funcione correctamente)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Crear y entrenar el modelo SVM con kernel RBF
modelo = SVC(kernel="rbf", C=1.0, gamma="scale")
modelo.fit(X_train_scaled, y_train)

# 6. Predecir
y_pred = modelo.predict(X_test_scaled)

# 7. Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 8. Mostrar resultados en consola
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

print("\nMatriz de Confusión (Texto):")
print(cm)

# 9. Visualizar la Matriz de Confusión de forma gráfica
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión (SVM)")
plt.show()