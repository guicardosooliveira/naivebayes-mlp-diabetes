from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train):
    """
    Treina e retorna um modelo Gaussian Naive Bayes.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model