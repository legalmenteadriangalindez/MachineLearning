import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score


#2separar datos 
X = df.drop("target", axis=1)

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)


#3 escalar variables 
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


#4 crear red neuronal 
model = MLPClassifier(

    hidden_layer_sizes=(32,16),

    activation="relu",

    solver="adam",

    alpha=0.0001,

    learning_rate_init=0.001,

    max_iter=500,

    random_state=42
)

#5 entrenar modelo
model.fit(X_train, y_train)

#6 predecir 
y_pred = model.predict(X_test)

#7 evaluar modelo
accuracy = accuracy_score(

    y_test,

    y_pred
)

print(accuracy)


#pipeline prfesional

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# Regresión

# Para problemas de regresión se utiliza MLPRegressor:
from sklearn.neural_network import MLPRegressor

regressor = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

