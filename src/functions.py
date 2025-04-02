# Imports 
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load the Data 
# Function that retrieves data from a provided path
# and reads the data as a pandas dataframe
def load_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file at {path} was not found.")
    if 'dev' in path:
        print('Data for development:')
    elif 'val' in path:
        print('Data for evaluation:')
    else:
        raise ValueError("The path provided does not contain data for development nor for evaluation.")
    data = pd.read_csv(path)
    return data

# codebase

## Pipeline for preprocessing

## Modeling functions 
models={
    'enet':ElasticNet(),
    'svr':SVR(),
    'breg':BayesianRidge()
}

### function to train the models
def train_model(model, x_train, x_test, y_train, y_test):
    results={}

    for model, model_instance in models.items():
        model_instance.fit(x_train, y_train)       
        y_pred=model_instance.predict(x_test)
        mse=mean_squared_error(y_test, y_pred)
        results[model]=mse
        print(f"{model} MSE: {mse:.4f}")
    return results

### function that  