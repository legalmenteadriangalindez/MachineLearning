import numpy as np

# Datos
x = 2.0
y = 1.0

# Parámetros
w = 0.5
b = 0.1

# Forward
z = w * x + b
a = 1 / (1 + np.exp(-z))

# Pérdida (MSE)
loss = 0.5 * (a - y) ** 2

# Backpropagation
dL_da = a - y
da_dz = a * (1 - a)
dz_dw = x

dL_dw = dL_da * da_dz * dz_dw

learning_rate = 0.1
w = w - learning_rate * dL_dw

print("Pérdida:", loss)
print("Gradiente:", dL_dw)
print("Nuevo peso:", w)