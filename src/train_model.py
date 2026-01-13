from sklearn.linear_model import LogisticRegression
import joblib
import os

def train(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/iris_model.pkl")

    return model
