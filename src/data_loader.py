import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/Iris.csv")

    df.drop("Id", axis=1, inplace=True)

    X = df.drop("Species", axis=1)
    y = df["Species"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
