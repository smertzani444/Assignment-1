import pandas as pd
import numpy as np
import joblib
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
import pprint
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import r_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

class Regressor:
    def __init__(self):
        self.models = {
            'enet': ElasticNet(),
            'svr': SVR(),
            'breg': BayesianRidge()
        }
        self.model_names = {
            'ElasticNet': ElasticNet(),
            'SVR': SVR(),
            'BayesianRidge': BayesianRidge()
        }
        self.param_grid = {
            'enet': {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8]
            },
            'svr': {
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1],
                'kernel': ['rbf', 'linear']
            },
            'breg': {
                'alpha_1': [1e-6, 1e-5],
                'lambda_1': [1e-6, 1e-5]
            }
        }

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

    def preprocess_data(self, df, output_path, columns_to_drop=[], scale=True):
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        num_list = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_list = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in cat_list:
            df[col] = LabelEncoder().fit_transform(df[col])

        if scale:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
        else:
            num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))])

        df[num_list] = num_pipeline.fit_transform(df[num_list])
        df.to_csv(output_path, index=False)
        print(f"The preprocessed data was saved to {output_path}.")
        return df
    
    def separate_features_target(self, df, target, columns_to_remove=None):
        if columns_to_remove is None:
            columns_to_remove=[]
        columns_to_remove=set(columns_to_remove + [target])
        X=df.drop(columns=[col for col in columns_to_remove if col in df.columns])
        y=df[target]
        return X, y
    

    def select_features(self, X, y, threshold=0.1):
        correlations = pd.Series(r_regression(X, y), index=X.columns)
        selected_features = correlations[correlations.abs() >= threshold].index.tolist()
        print(f"The selected features of {X.shape[1]} were: {len(selected_features)}")
        return selected_features, correlations

    def train_models(self, X, y):
        results = {}

        x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
        
        for name, model in self.models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            rmse = root_mean_squared_error(y_test, y_pred)
            results[name] = rmse
            print(f"{name} RMSE: {rmse:.4f}")
        return results

    def generate_param_combintions(self, param_grid):
        model_combinations = {
        model: [
            dict(zip(params.keys(), values))
            for values in itertools.product(*params.values())
        ]
        for model, params in param_grid.items()
        }
        return model_combinations

    def model_tuning(self, param_grid, X, y, cv=5):
        best_results = {}
        for model, param_grid in param_grid.items():
            best_rmse = float('inf')
            best_model = None
            best_params = None
            for combo in itertools.product(*param_grid.values()):
                combo_dict = dict(zip(param_grid.keys(), combo))
                model_instance = self.models[model].__class__(**combo_dict)
                scores = cross_val_score(model_instance, X, y,
                                         scoring='neg_root_mean_squared_error', cv=cv)
                rmse = (-scores.mean())**0.5
                print(f"[{model}] Tested params: {combo_dict}")
                print(f"[{model}] RMSE: {rmse:.4f}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_instance
                    best_params = combo_dict
                    print(f"[{model}] New best RMSE: {best_rmse:.4f}")
                    print(f"[{model}] Best params so far: {best_params}")
            best_results[model] = {
                'Best RMSE': best_rmse,
                'Best Model': best_model,
                'Best Params': best_params
            }
        return best_results

    def summarize(self, scores):
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        ci95 = t.interval(0.95, len(scores) - 1, loc=mean, scale=std / np.sqrt(len(scores)))
        return {
            'mean': mean,
            'median': np.median(scores),
            '95% CI': ci95
        }

    
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

        best_rmse = float('inf')
        best_model = None

        for i in range(runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics['rmse'].append(rmse)
            metrics['mae'].append(mae)
            metrics['r2'].append(r2)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = joblib.loads(joblib.dumps(model))  # Deep copy of best model

        summary = {k: self.summarize(v) for k, v in metrics.items()}

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:  # Only create directory if one is specified
                os.makedirs(save_dir, exist_ok=True)
            joblib.dump(best_model, save_path)
            print(f"Best model saved to {save_path}")

  
        for metric, values in metrics.items():
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=values)
            plt.title(f"{metric.upper()} Distribution")
            plt.ylabel(metric.upper())
            plt.show()

        return summary, metrics


    
