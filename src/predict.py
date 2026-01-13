import joblib
import os

def predict_species(features):
    model = joblib.load("models/iris_model.pkl")
    prediction = model.predict([features])[0]

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sample_prediction.txt", "w") as f:
        f.write(f"Input Features: {features}\n")
        f.write(f"Predicted Species: {prediction}\n")

    return prediction
