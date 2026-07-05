import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset desde un archivo CSV
# Reemplaza 'datos_salarios.csv' por la ruta real de tu archivo
df = pd.read_csv("datos_salarios.csv")

# 2. Definir variables
X = df[["edad", "experiencia", "educación"]]  # Variables predictoras
y = df["salario"]  # Variable objetivo

# 3. Dividir en entrenamiento y prueba (70% - 30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 4. Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5. Predecir
y_pred = modelo.predict(X_test)

# 6. Calcular métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 7. Mostrar resultados
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")