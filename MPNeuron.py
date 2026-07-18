import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ===========================
# Clase MP Neuron
# ===========================
class MPNeuron:

    def __init__(self):
        self.threshold = None

    def model(self, x):
        return sum(x) >= self.threshold

    def predict(self, X):
        Y = []
        for x in X:
            Y.append(self.model(x))
        return np.array(Y)

    def fit(self, X, Y):

        accuracy = {}

        # Buscar el mejor threshold
        for th in range(X.shape[1] + 1):

            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y, Y_pred)

        self.threshold = max(accuracy, key=accuracy.get)

        print(f"Mejor threshold: {self.threshold}")
        print(f"Accuracy entrenamiento: {accuracy[self.threshold]:.4f}")


data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(X.head())
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# Mediana calculada únicamente sobre entrenamiento
thresholds = X_train.median()

X_train_bin = (X_train >= thresholds).astype(int)
X_test_bin = (X_test >= thresholds).astype(int)


mp_neuron = MPNeuron()

mp_neuron.fit(X_train_bin.values, y_train.values)

y_pred = mp_neuron.predict(X_test_bin.values)


acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

print("\nMatriz de confusión")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación")
print(classification_report(y_test, y_pred))