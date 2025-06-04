from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def train_naive_bayes(X_train, y_train):
    """
    Treina e retorna um modelo Gaussian Naive Bayes.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_mlp(x_train, y_train):
    model_mlp = MLPClassifier(max_iter=1000, random_state=42)

    grid_search = get_best_hiperparams(model_mlp)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_


def get_best_hiperparams(model):
    param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (20, 10)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    return grid_search