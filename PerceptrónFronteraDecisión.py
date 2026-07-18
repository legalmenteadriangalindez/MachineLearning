import numpy as np

# ==============================
# DATASET OR
# ==============================

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 1])

# ==============================
# PARÁMETROS INICIALES
# ==============================

w = np.zeros(2)
b = 0

eta = 1
epochs = 10

# ==============================
# FUNCIÓN ESCALÓN
# ==============================

def step(z):

    if z >= 0:
        return 1

    return 0

# ==============================
# ENTRENAMIENTO
# ==============================

for epoch in range(epochs):

    print(f"\nÉPOCA {epoch+1}")

    for i in range(len(X)):

        # Ejemplo actual
        x = X[i]

        # Valor esperado
        target = y[i]

        # Forward Propagation
        z = np.dot(w, x) + b

        # Predicción
        prediction = step(z)

        # Error
        error = target - prediction

        # Actualización de pesos
        w = w + eta * error * x

        # Actualización del bias
        b = b + eta * error

        print("---------------------")
        print("Entrada:", x)
        print("Esperado:", target)
        print("Predicción:", prediction)
        print("Error:", error)
        print("Pesos:", w)
        print("Bias:", b)


# Visualizar la frontera de decisión
import matplotlib.pyplot as plt

# Dibujar puntos de cada clase
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i,0], X[i,1], marker="o", s=100, label="Clase 0" if i == 0 else "")
    else:
        plt.scatter(X[i,0], X[i,1], marker="x", s=100, label="Clase 1" if i == 1 else "")

# Valores de x para dibujar la recta
x_values = np.linspace(-0.5, 1.5, 100)

# Ecuación de la frontera:
# w1*x1 + w2*x2 + b = 0
# Despejando x2:
# x2 = -(w1*x1 + b) / w2

if w[1] != 0:
    y_values = -(w[0] * x_values + b) / w[1]
    plt.plot(x_values, y_values, label="Frontera de decisión")

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()