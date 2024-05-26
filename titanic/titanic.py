# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore


def get_data(path):
    '''
    This function gets the required data from a specified path.
    Parameters:
    - path: the filepath of the csv file
    Returns: Data from the csv file.
    '''
    # Get the needed data.
    df = pd.read_csv(path)
    return df


def split_data(df, X_drop_cols, y_cols):
    '''
    This function splits the data into X and y values
    Parameters:
    - df: The data you want to split
    - X_drop_cols: The columns you want to drop in the X data
    - y_cols: The columns you want to keep in the y data.
    Returns:
    - splitted X and y values.
    '''
    # Get the correct columns
    X = df.drop(X_drop_cols, axis=1)
    y = df[y_cols]
    return X, y


def train_score_model(model, X_train, X_test, y_train, y_test):
    '''
    This function trains the model and after that it gives the score back.
    Parameters:
    - model: The model which you want to train
    - X_train: The X data to train the model with
    - X_test: The X data to test the model with
    - y_train: The y data to train the model with
    - y_test: The
    '''
    # Train the model.
    model.fit(X_train, y_train)
    # Predict using the LogisticRegression model.
    return model, model.score(X_test, y_test)


path = "data/titanic.csv"

df = get_data(path)

# Split the data
X, y = split_data(df, ["Survived", "Name"], "Survived")

# Initialize LabelEncoder
label_encoder = LabelEncoder()
X["Sex"] = label_encoder.fit_transform(X["Sex"])

# Split the X and y values into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initiate LogisticRegression model
lr = LogisticRegression()

lr, score = train_score_model(lr, X_train, X_test, y_train, y_test)
print(score)

def trailingspace():
    print("Hello World!")
