import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Leer el archivo CSV
df = pd.read_csv('ejemplo.csv')

# 2. Extraer las variables (asegurar que X sea 2D para sklearn)
X = df[['X']].values          # shape (n, 1)
y = df['Y'].values            # shape (n,)

# 3. Dividir en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5. Predecir sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# 6. Calcular métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 7. Mostrar resultados
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)