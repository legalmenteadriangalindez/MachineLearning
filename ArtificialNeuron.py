import numpy as np

# Entradas
x = np.array([4, 2])

# Pesos
w = np.array([0.8, -0.5])

# Sesgo
b = 1

# Combinación lineal
z = np.dot(x, w) + b

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Salida
a = sigmoid(z)
  
print("z =", z)
print("Salida =", a)