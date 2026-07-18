from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

modelo = MLPClassifier(max_iter=500, random_state=42)

param_grid = {
    "hidden_layer_sizes": [(32,), (64,), (64, 32)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01]
}

grid = GridSearchCV(
    estimator=modelo,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)