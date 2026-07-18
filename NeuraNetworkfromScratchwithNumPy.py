import numpy as np

# ============================
# Datos de entrenamiento (XOR)
# ============================

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

# ============================
# Inicialización
# ============================

np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# ============================
# Funciones de activación
# ============================

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ============================
# Hiperparámetros
# ============================

learning_rate = 0.1
epochs = 10000

# ============================
# Entrenamiento
# ============================

for epoch in range(epochs):

    # Forward

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    # Binary Cross Entropy

    loss = -np.mean(
        y*np.log(A2+1e-8) +
        (1-y)*np.log(1-A2+1e-8)
    )

    # Backpropagation

    m = len(X)

    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Descenso por gradiente

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        print(epoch, loss)

# ============================
# Predicciones
# ============================

pred = sigmoid(relu(X @ W1 + b1) @ W2 + b2)

print(pred)