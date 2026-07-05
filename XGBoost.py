import xgboost as xgb
import pandas as pd  # <--- Agregado para leer el CSV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar los datos desde el archivo CSV
# Cambia 'tus_datos.csv' por el nombre real de tu archivo
df = pd.read_csv('tus_datos.csv')

# Seleccionar variables predictoras y objetivo (ajusta los nombres según tu CSV)
# Nota: Al ser XGBRegressor, asumimos que 'clase' o la meta es un valor numérico continuo
X = df[['manzana', 'naranja']]   # Variables predictoras
y = df['clase']                  # Variable objetivo (numérica continua)

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Crear el modelo (Sin StandardScaler, directo a XGBoost)
modelo = xgb.XGBRegressor(
    n_estimators=100,      # Número de árboles
    max_depth=6,           # Profundidad máxima
    learning_rate=0.1,     # Tasa de aprendizaje (eta)
    subsample=0.8,         # Muestreo de filas
    colsample_bytree=0.8,  # Muestreo de columnas
    reg_alpha=0,           # Regularización L1
    reg_lambda=1,          # Regularización L2
    random_state=42
)

# 4. Entrenar el modelo
modelo.fit(X_train, y_train)

# 5. Predecir
y_pred = modelo.predict(X_test)

# 6. Calcular métricas de regresión
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")