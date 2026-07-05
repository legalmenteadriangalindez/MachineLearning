import numpy as np
import pandas as pd

# 1. Cargar los datos desde el archivo CSV
# Cambia 'tus_datos.csv' por el nombre real de tu archivo
df = pd.read_csv('tus_datos.csv')

# NOTA: Como ya estás cargando tu propio archivo CSV, 
# se elimina el bloque de 'make_classification' y la creación manual del DataFrame.

# 2. Estadísticas descriptivas
print("=== Descripción ===")
print(df.describe())

# 3. Valores faltantes
print("\n=== Valores Faltantes ===")
print(df.isnull().sum())

# 4. Filas duplicadas
print("\n=== Duplicados ===")
print(f"Filas duplicadas: {df.duplicated().sum()}")

# 5. Matriz de correlación
print("\n=== Matriz de Correlación ===")
print(df.corr(numeric_only=True))

# 6. Skewness (asimetría)
print("\n=== Skewness ===")
print(df.skew(numeric_only=True))

# 7. Kurtosis (curtosis)
print("\n=== Kurtosis ===")
print(df.kurtosis(numeric_only=True))