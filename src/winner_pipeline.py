import pandas as pd
import numpy as np
import joblib
import sys
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
import pprint
import copy
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import r_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

class BregWModel:
    def __init__(self):
        self.model=BayesianRidge(lambda_1=1e-05)
        self.metadata_columns_to_drop=['Unnamed: 0', 'Project ID', 'Experiment type', 'Disease MESH ID']
        self.columns_to_remove=['Sex', 'Host age']
        self.target='BMI'
        self.selcted_features=['Alistipes putredinis', 'Anaerotruncus colihominis', 'Bacillus megaterium', 'Bacteroides massiliensis', 'Bifidobacterium saguini', 'Christensenella minuta', 'Clostridium amylolyticum', 'Desulfonispora thiosulfatigenes', 'Desulfovibrio desulfuricans', 'Lachnospiraceae bacterium 7_1_58FAA', 'Oscillibacter valericigenes', 'Papillibacter cinnamivorans', 'Parabacteroides johnsonii', 'Pseudoflavonifractor capillosus', 'Ruminiclostridium thermocellum', 'Ruminococcus albus', 'Ruminococcus champanellensis', 'Ruminococcus flavefaciens', 'Sporobacter termitidis', 'Bacteroides pectinophilus', 'Clostridium asparagiforme', 'Clostridium clariflavum', 'Clostridium colinum', 'Clostridium propionicum', 'Clostridium stercorarium', 'Clostridium symbiosum', 'Eubacterium brachy', 'Eubacterium dolichum', 'Eubacterium sulci']

    def load_data(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The file at {path} was not found.")
        if 'dev' in path:
            print('Data for development:')
        elif 'val' in path:
            print('Data for evaluation:')
        else:
            raise ValueError("The path provided does not contain data for development nor for evaluation.")
        return pd.read_csv(path)

    def preprocess_data(self, df, output_path, metadata_columns_to_drop=['Unnamed: 0', 'Project ID', 'Experiment type', 'Disease MESH ID']):
        df = df.drop(columns=[col for col in self.metadata_columns_to_drop if col in df.columns], errors='ignore')

        if hasattr(self, 'selected_features') and self.selected_features is not None:
            df = df[[col for col in df.columns if col in self.selected_features]] 
        
        num_list = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_list = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in cat_list:
            df[col] = LabelEncoder().fit_transform(df[col])
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
        df[num_list] = num_pipeline.fit_transform(df[num_list])
        df.to_csv(output_path, index=False)
        print(f"The preprocessed data was saved to {output_path}.")
        return df                                                                                                                         # returns reduced df 700x137
    
    def train_model(self, df, target, columns_to_remove=None, selected_features=None):
        if selected_features is not None: 
            selected_features = [col for col in selected_features if col in df.columns]
            selected_df = df[selected_features + [target]]

        if columns_to_remove is None:
            columns_to_remove=[]
        columns_to_remove=set(columns_to_remove + [target])
        X=selected_df.drop(columns=[col for col in columns_to_remove if col in selected_df.columns])
        y=selected_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)                                                      # splits X and y into training and test sets 

        scaler=StandardScaler()
        model = BayesianRidge(lambda_1=1e-05)
        X_train = scaler.fit_transform(X_train)
        model.fit(X_train, y_train)
        X_test = scaler.fit_transform(X_test)  # Use transform, not fit_transform!
        y_pred = model.predict(X_test)


        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"Model: BayesianRidge RMSE: {rmse:.4f}")
        return selected_df, rmse                                                                   # selected df should be 700x30
    
    def align_evaluation_set(self, dev_df, val_df):
        dev_columns = dev_df.columns
        val_aligned = val_df.copy()
        val_aligned = val_aligned.reindex(columns=dev_columns, fill_value=0)
        print("Evaluation dataset was aligned to development feature set.")
        return val_aligned

    def evaluate_model(self, model, X, y, runs=30, test_size=0.2, save_path=None):
        metrics = {
            'rmse': [],
            'mae': [],
            'r2': []
        }

        for i in range(runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            scaler=StandardScaler()
            model=BayesianRidge(lambda_1=1e-05)
            X_train=scaler.fit_transform(X_train)
            model.fit(X_train, y_train)
            X_test=scaler.fit_transform(X_test)
            y_pred = model.predict(X_test)
             
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics['rmse'].append(rmse)
            metrics['mae'].append(mae)
            metrics['r2'].append(r2)


        results = {k: self.summarize(v) for k, v in metrics.items()}

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(BayesianRidge(lambda_1=1e-05), save_path)
            print(f"Best model saved to {save_path}")

        for metric, values in metrics.items():
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=values)
            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())
            plt.show()

        return results 






        




        

