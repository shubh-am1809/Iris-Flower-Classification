from src.data_loader import load_data
from src.train_model import train
from src.evaluate_model import evaluate
from src.predict import predict_species

def main():
    X_train, X_test, y_train, y_test = load_data()

    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)

    sample = [5.1, 3.5, 1.4, 0.2]
    result = predict_species(sample)

    print("Predicted Species:", result)
    print("All outputs saved in outputs/ folder")

if __name__ == "__main__":
    main()
