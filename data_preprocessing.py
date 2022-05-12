import numpy as np
import pandas as pd
import pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = pathlib.Path("BankNote_Authentication.csv")

def preprocess():
    data_headers = ['variance','skewness','kurtosis','entropy','class']
    dataset = pd.read_csv(DATASET_PATH)
    print(dataset.head(10))

    x = dataset.drop(["class"],axis=1) # data
    y = dataset['class'] # label
    # standaridization of data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # split data into training and testing sets with the ratio of 8:2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    return x_train, x_test, y_train, y_test


