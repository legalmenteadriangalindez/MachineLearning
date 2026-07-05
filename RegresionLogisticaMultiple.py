import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# 1. Cargar datos
df = pd.read_csv("clients.csv")

# 2. Exploración inicial
print("Head:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())

# 3. Matriz de correlación y Heatmap
corr = df.corr(numeric_only=True)
print("\nMatriz de correlación:")
print(corr)

plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()

# 4. Definir variables
X = df[["edad", "ingreso", "visitas"]]  # Predictoras
y = df["compra"]  # Variable objetivo (0/1)

# 5. Dividir en entrenamiento y prueba (70% - 30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 6. Crear y entrenar el modelo de Regresión Logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 7. Predicciones y Probabilidades
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)  # Probabilidades para cada clase

print("\nProbabilidades (primeras 5):")
print(y_prob[:5])

# 8. Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# 9. Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# 10. Reporte de clasificación detallado
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Coeficientes del modelo y Odds Ratio
coeficientes = pd.DataFrame({"Variable": X.columns, "Beta": modelo.coef_[0]})

# Calcular Odds Ratio (exponencial de los coeficientes)
coeficientes["Odds_Ratio"] = np.exp(coeficientes["Beta"])

print("\nCoeficientes y Odds Ratio:")
print(coeficientes)

# Intercepto
print(f"\nIntercepto: {modelo.intercept_[0]:.4f}")