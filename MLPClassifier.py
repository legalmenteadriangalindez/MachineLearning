import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# Importamos métricas adicionales para una evaluación completa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Cargar el conjunto de datos desde un archivo CSV
# Reemplaza 'tu_archivo.csv' por la ruta de tu archivo y 'target' por el nombre de tu columna objetivo
df = pd.read_csv('tu_archivo.csv')

# Suponiendo que la última columna es la variable a predecir (y) y el resto son las características (X)
X = df.drop(columns=['target'])  # Cambia 'target' por el nombre real de tu columna de etiquetas
y = df['target']

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Crear el modelo MLP
model = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

# 4. Entrenar el modelo
model.fit(X_train, y_train)

# 5. Predicciones
y_pred = model.predict(X_test)

# 6. Evaluación detallada
print("=== Métricas de Evaluación ===")
print(f"Accuracy General: {accuracy_score(y_test, y_pred):.4f}\n")

print("Reporte de Clasificación (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred))

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))