from distutils.log import error
import numpy as np
import pandas as pd
import pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BANKNOTE_PATH = "BankNote_Authentication.csv"
EMAILSPAM_PATH = "email_spam.csv"

def preprocess(dataset_dir_path, dataset_name):
    dataset_dir_path = pathlib.Path(dataset_dir_path)
        
    if dataset_name == "bank":
        data_headers = ['variance','skewness','kurtosis','entropy','class']
        dataset = pd.read_csv(dataset_dir_path/BANKNOTE_PATH)

        x = dataset.drop(["class"],axis=1) # data
        y = dataset['class'] # label
        # standardiz data
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        # split data into training and testing sets with the ratio of 8:2
    elif dataset_name == "email":
        dataset = pd.read_csv(dataset_dir_path/EMAILSPAM_PATH)
        # remote index
        dataset = dataset.iloc[: , 1:]
        keywords = ['yes', 'no', 'HTML', 'Plain', 'none', 'big', 'small']
        numerical_value = [1,0,0,1,0,1,2]
        dataset = dataset.replace(keywords,numerical_value)
        y = dataset["spam"]
        x = dataset.drop(["spam"],axis=1)
        # standardiz data
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        raise Exception("Dataset name has to be one of [bank, email]")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    return x_train, x_test, y_train, y_test


