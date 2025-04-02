# Imports 
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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


# Pipeline for preprocessing
# Function that:
# Drops specific columns,
# handles missing values, 
# encodes categorical features and
# saves preprocessed data in a csv file in the output path that we provide
def preprocess_data(df, output_path, columns_to_drop, scale=True):
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    num_list = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_list = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    for column in num_list + cat_list:
        if column not in df.columns:
            raise ValueError(f"'{column}' could not be found in the dataframe provided.") 
    
    for col in cat_list:
        df[col] = LabelEncoder().fit_transform(df[col])

    if scale:
        num_pipeline=Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    else:
        num_pipeline=Pipeline([('imputer', SimpleImputer(strategy='mean'))])


    df[num_list] = num_pipeline.fit_transform(df[num_list])

    df.to_csv(output_path, index=False)

    print(f"The preprocessed data was saved to {output_path}.")
    return df

# Function that selects features based on the Pearson's Correlation Coefficient Method
def select_features(X, y, threshold):
    correlations=pd.Series(r_regression(X, y), index=X.columns)
    selected_features=correlations[correlations.abs() >= threshold].index
    reduced_df=X[selected_features]
    
    print(f"The selected features of {X.shape[1]} were: {len(selected_features)}")
    return reduced_df, correlations

X=dev_set_cleaned_df.drop(['BMI', 'Host age', 'Sex'], errors='ignore') # Bacteria Species -> features
y=dev_set_cleaned_df['BMI']

##Modeling functions 
models={
    'enet':ElasticNet(),
    'svr':SVR(),
    'breg':BayesianRidge()
}

# Function to train the models
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