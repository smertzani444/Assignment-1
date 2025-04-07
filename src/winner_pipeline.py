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

class BMI_predictor:
    def __init__(self):
        self.model=BayesianRidge(lambda_1=1e-05)
        self.metadata_columns_to_drop=['Unnamed: 0', 'Project ID', 'Experiment type', 'Disease MESH ID']
        self.pipeline=None
        self.target='BMI'
        self.selected_features=['Alistipes putredinis', 'Anaerotruncus colihominis', 'Bacillus megaterium', 
                                'Bacteroides massiliensis', 'Bifidobacterium saguini', 'Christensenella minuta',
                                'Clostridium amylolyticum', 'Desulfonispora thiosulfatigenes', 'Desulfovibrio desulfuricans', 
                                'Lachnospiraceae bacterium 7_1_58FAA', 'Oscillibacter valericigenes', 'Papillibacter cinnamivorans', 
                                'Parabacteroides johnsonii', 'Pseudoflavonifractor capillosus', 'Ruminiclostridium thermocellum', 
                                'Ruminococcus albus', 'Ruminococcus champanellensis', 'Ruminococcus flavefaciens', 'Sporobacter termitidis', 
                                'Bacteroides pectinophilus', 'Clostridium asparagiforme', 'Clostridium clariflavum', 'Clostridium colinum', 
                                'Clostridium propionicum', 'Clostridium stercorarium', 'Clostridium symbiosum', 'Eubacterium brachy', 
                                'Eubacterium dolichum', 'Eubacterium sulci']
    
    # Function for loading the data, only argument is the path in which the data can be found
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

    # function that preprocesses the data    
    def preprocess_pipeline(self, df, scale=False):
        df = df.drop(columns=[col for col in self.metadata_columns_to_drop if col in df.columns])  # Drops the metadata columns that were user defined
        
        if hasattr(self, 'selected_features') and self.selected_features is not None:
            df = df[[col for col in df.columns if col in self.selected_features]]                  # Drops everything but the selected features

        num_feats = df.select_dtypes(include=[np.number]).columns.tolist()                         # df containing the columns with numerical features
        cat_feats = df.select_dtypes(exclude=[np.number]).columns.tolist()                         # df containing the columns with categorical features 
        
        if scale:                                                                                  # Applys scaling whenever is necessary 
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
        else:
            num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))])                 # pipeline for handling missing values 
        cat_pipeline = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])             # pipeline for encoding categorical values

        preprocess = ColumnTransformer([                                                           # applies pipeline to the features 
            ('num', num_pipeline, num_feats),
            ('cat', cat_pipeline, cat_feats)
        ])

        self.pipeline = Pipeline([
            ('preprocessing', preprocess),
            ('model', self.model)
        ])
        return df                                                                                  # returns reduced df (700x137)
    
    # function that separates features and target,
    # 
    def train_model(self, df, target='BMI', columns_to_drop=None, scale=True, selected_features=[]):
        if columns_to_drop is None:                                                                # creates empty list for columns to drop if not provided with one
            columns_to_drop=[]
    
        if self.selected_features is not None:
            selected_df = df[[col for col in self.selected_features if col in df.columns] + [target]]
        else:
            selected_df = df.copy()
                                                                                                   # creates df that contains only the selected features 
                                                                                                   # and the target

        X = selected_df.drop(columns=[target])                                                              # X=> only the features (29 columns)
        y = selected_df[target]                                                                             # y=> only the target   (1 column)

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)                                                      # splits X and y into training and test sets 

    
        self.preprocess_pipeline(X_train)                                                             # preprocessing X_train, includes scaling  
        self.pipeline.fit(X_train, y_train)                                                        # before training the model in order to avoid data leakage
        self.preprocess_pipeline(X_test)                                                              # then preprocessing X_test
        y_pred = self.pipeline.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"Model: BayesianRidge RMSE: {rmse:.4f}")
        return selected_df, rmse                                                                  # returns the rmse score and the selected_df 
                                                                                                  # which contains the selected features and the target column
                                                                                                  # selected_df (700x30)

    
    def align_evaluation_set(self, dev_df, val_df):                                               # aligns val_df to the selected_df 
        dev_columns = dev_df.columns                                                              # whichever is the val_df this function returns an evaluation df that contains the same columns as the selected_df 
        val_aligned = val_df.copy()
        val_aligned = val_aligned.reindex(columns=dev_columns, fill_value=0)
        print("Evaluation dataset was aligned to development feature set.")
        return val_aligned
    
    def evaluate_model(self, df, target_column='BMI', drop_columns=None, scale=False):
        metrics = {                                                                              # evaluation metrics 
            'rmse': [],
            'mae': [],
            'r2': []
        }

        if drop_columns is None:
            drop_columns = []
        drop_columns = set(drop_columns + [target_column])
        
        X = df.drop(columns=[target_column])                                                    # separates features and target 
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)                                               # splits X and y into tarining and test sets

        self.preprocess_pipeline(X_train)                                                          # preprocessing and scaling X_train
        self.pipeline.fit(X_train, y_train)                                                     # applying breg model
        self.preprocess_pipeline(X_test)                                                           # preprocessing and scaling X_test
        y_pred = self.pipeline.predict(X_test)                                                  # making predictions 

        rmse = root_mean_squared_error(y_test, y_pred)                                          # computing the evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Evaluation Metrics:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}")

        for metric, values in metrics.items():                                                   # creates boxplots for each of the metric scores
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=values)
            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())
            plt.show()

        return {"RMSE": rmse, "MAE": mae, "R2": r2}
    
        

    def save_model(self, save_path='final_models/winner.pkl'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.pipeline, save_path)
        print(f"Final model saved to {save_path}")

        




        

