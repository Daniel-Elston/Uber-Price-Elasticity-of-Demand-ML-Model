import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import tqdm
import timeit
import textwrap
import sklearn
import xgboost

import pytest

from scipy import stats
from scipy.stats import zscore
from scipy.optimize import curve_fit

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


from sklearn.model_selection import GridSearchCV


pd.options.mode.chained_assignment = None

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


def file_load(input_files):
        
    for i, file in enumerate(input_files):
        print(file)
        globals()[f'df{i+1}'] = pd.read_csv(input_files[i])
    print(f'Total number of files loaded: {len(input_files)}')
    df_store = [eval(f'df{i+1}') for i in range(len(input_files))]
    
    return df_store

def trim_raw_data(df, required_cols, df_frac):
    df = df[required_cols]
    df = df.sample(frac=df_frac)
    return df



class CreateDataDictionary:

    def __init__(self, df):
        '''This class provides functions to quickly develop a data dictionary for your data set'''
        self.df = df
        return None

    def make_my_data_dictionary(self, dataFrame):
        '''Create an initial data dictionary excluding definitions for meaning of features'''

        df_cols = dataFrame.columns
        df_DataDict = {}

        for col in df_cols:
                df_DataDict[col] = {
                               'Dtype': str(self.df.dtypes[col]),
                               'Expected_Type': str(''),
                               'Length': len(self.df[col]),
                               'Null_Count': sum(self.df[col].isna()),
                               'Size(Memory)': self.df.memory_usage()[col],
                               'Definition': str(''),
                               'Range': (self.df[col].min(), self.df[col].max())
                                }

        df_DD = pd.DataFrame(df_DataDict)

        return df_DD

    def define_data_meaning(self, df_data_dictionary):
        '''Quickly provide input regarding each columns meaning and transpose into a usable dictionary'''

        df_cols = df_data_dictionary.columns
        d = 'Definition'

        for col in df_cols:
            df_data_dictionary[col][d] = input('Provide a data definition for {}'.format(col))

        df_data_dictionary = df_data_dictionary.transpose()

        return df_data_dictionary

    def update_dd_definition(self, df_data_dictionary, attribute):
        try:
            df_dd = df_data_dictionary.transpose()
            df_dd[attribute]['Definition'] = input('Provide a data definition for {}'.format(attribute))
            return df_dd
        except:
            print('Sorry, there was an error.  Check attribute name and try again')



class Dataset:
    def __init__(self, data, threshold):
        self.data = data
        self.threshold = threshold
    
    def data_transform(self):
        self.data['pickup_datetime'] = pd.to_datetime(self.data['pickup_datetime']).dt.tz_localize(None)
        self.data['pickup_datetime'] = self.data['pickup_datetime'].dt.floor('H')
        
        self.data['Label_Hour'] = self.data['pickup_datetime'].dt.hour
        self.data['Label_Date'] = self.data['pickup_datetime'].dt.date

        self.data['day_name'] = pd.to_datetime(self.data['pickup_datetime']).dt.day_name()
        self.data['day_of_the_week'] = pd.to_datetime(self.data['pickup_datetime']).dt.weekday
        return self.data
            
    def data_remove_outliers(self):
        self.data = self.data[(np.abs(stats.zscore(self.data['passenger_count'])) < 2)]
        self.data = self.data[(np.abs(stats.zscore(self.data['fare_amount'])) < 2)]
        return self.data

    def data_clean(self):
        self.data['fare_amount'] = abs(self.data.fare_amount)
        self.data.passenger_count.replace(0, 1, inplace=True)
        self.data.fare_amount.replace(0, 1, inplace=True)
        self.data = self.data.dropna()
        return self.data
    
    def winsorization(self):
        median = self.data['sPED'].median()
        lower_quantile = self.data['sPED'].quantile(self.threshold)
        upper_quantile = self.data['sPED'].quantile(1-self.threshold)
        self.data['sPED'][self.data['sPED'] < lower_quantile] = -1*median
        self.data['sPED'][self.data['sPED'] > upper_quantile] = median
        return self.data
    
    def index_set(self):
        self.data = self.data.set_index('pickup_datetime')
        self.data.index = self.data.index.strftime('%Y-%m-%d-%H')
        return self.data

class DataGrouper:
    def __init__(self, data, col_group):
        self.data = data
        self.col_group = col_group
    
    def group_by_hour(self, col_group):
        self.data = self.data[col_group].groupby(['pickup_datetime'])['passenger_count','fare_amount'].agg({'passenger_count':'mean','fare_amount':'mean'}).reset_index()
        return self.data
    
    def group_by_day(self, col_group):
        self.data = self.data[col_group].groupby(['Label_Date'])['passenger_count','fare_amount'].agg({'passenger_count':'mean','fare_amount':'mean'}).reset_index()
        return self.data
    
class FeatureExtraction:
    def __init__(self, data, col_feature, n_window):
        self.data = data
        self.col_feature = col_feature
        self.n_window = n_window

    def calc_pct_change(self, col_feature):
        for col in col_feature:
            self.data[col + '_pct_change'] = self.data[col].pct_change()
        return self.data
    
    def calc_sma(self, col, n_window):
        self.data[col + '_sma'] = self.data[col].rolling(window=self.n_window).mean()
        return self.data
        
    def calc_ema(self, col, n_window):
        self.data[col + '_ema'] = self.data[col].ewm(span=n_window, adjust=False).mean()
        return self.data
        
    def calc_sma_PED(self):
        self.data['sPED'] = self.data.passenger_count_pct_change_sma / self.data.fare_amount_pct_change_sma
        return self.data
        
    def calc_ema_PED(self):
        self.data['ePED'] = self.data.passenger_count_pct_change_ema / self.data.fare_amount_pct_change_ema
        return self.data

class ModelSelection:
    def __init__(self, data, target_column, cv_splits, test_split):
        self.data = data
        self.target_column = target_column
        self.cv_splits = cv_splits
        self.test_split = test_split
        # self.models = [DecisionTreeRegressor(random_state=0), LinearRegression(),
        #                RandomForestRegressor(), 
        #                SVR(C=1.0, epsilon=0.2), 
        #                xgboost.XGBRegressor(n_estimators=100, max_depth=5, eta=0.1, subsample=1-test_split)]
        self.models = {
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(), 
            'SVR': SVR(C=1.0, epsilon=0.2), 
            'XGBRegressor': xgboost.XGBRegressor(n_estimators=100, max_depth=5, eta=0.1, subsample=1-test_split)
            }
        
    def split_dataset(self):
        input_features = self.data.drop(self.target_column, axis=1)
        target_variable = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(input_features, target_variable, shuffle=False, test_size=self.test_split)
        return X_train, X_test, y_train, y_test
    
    def model_evaluation(self, model, X_train, y_train, cv_splits):
        std_clf = make_pipeline(StandardScaler(), model)

        r2_scores = []
        mae_scores = []
        rmse_scores = []

        kf = KFold(n_splits=cv_splits, shuffle=False)
        for train_index, test_index in kf.split(X_train):
            X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

            std_clf.fit(X_train_cv, y_train_cv)
            y_pred = std_clf.predict(X_test_cv)

            r2_scores.append(r2_score(y_test_cv, y_pred))
            mae_scores.append(mean_absolute_error(y_test_cv, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test_cv, y_pred)))

        return {"Mean R^2": sum(r2_scores) / cv_splits, 
                "Mean MAE": sum(mae_scores) / cv_splits,
                "Mean RMSE": sum(rmse_scores) / cv_splits
                }
        
    def run_model_evaluation(self):
        X_train, X_test, y_train, y_test = self.split_dataset()
        results = {}
        for model_name, model in self.models.items():
            results[model_name] = self.model_evaluation(model, X_train, y_train, self.cv_splits)
        return results
    
    def select_initial_model(self):
        X_train, X_test, y_train, y_test = self.split_dataset()
        best_model = None
        best_score = 0
        for model_name, model in self.models.items():
            # Define hyperparameters for grid search
            if model_name == 'DecisionTreeRegressor':
                params = {'max_depth': [5, 10, 15]}
            elif model_name == 'LinearRegression':
                params = {'fit_intercept': [True, False]}
            elif model_name == 'RandomForestRegressor':
                params = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15]}
            elif model_name == 'SVR':
                params = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
            elif model_name == 'XGBRegressor':
                params = {'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 150]}
                
            # Perform grid search
            grid = GridSearchCV(estimator=model, param_grid=params, cv=self.cv_splits, n_jobs=-1)
            grid.fit(X_train, y_train)
            
            # Determine if this model is the best so far
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                best_params = grid.best_params_
        
        best_model_name = type(best_model).__name__
        if best_model_name == 'XGBRegressor':
            model_class = xgboost.XGBRegressor
        else:
            model_class = getattr(sklearn.ensemble, best_model_name)
        selected_model = model_class(**best_params)
        
        print('\n')
        print("Selected model: {}".format(type(best_model).__name__))
        print("Hyperparameters: {}".format(best_params))
        print("Model score: {}".format(best_score))
        print('\n')

        return selected_model
            
class ModelTrainer:
    def __init__(self, data, X_train, y_train, selected_model, cv_splits):
        self.data = data
        self.X_train = X_train
        self.y_train = y_train
        self.selected_model = selected_model
        self.cv_splits = cv_splits
    
    def train_model(self):
        
        # create a pipeline that includes scaling and the selected model
        std_clf = make_pipeline(StandardScaler(), self.selected_model)
            
        # initialize KFold object
        kf = KFold(n_splits=self.cv_splits, shuffle=False)
        
        train_results = []
        
        # loop through each fold of the data
        for train_index, test_index in kf.split(self.X_train):
            
            # split data into train and test sets for this fold
            X_train_cv, X_test_cv = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
            y_train_cv, y_test_cv = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
            
            # fit the pipeline on the training data 
            std_clf.fit(X_train_cv, y_train_cv)
            
            # calculate the score on the training data
            apply = std_clf.score(X_train_cv, y_train_cv)
            train_results.append(apply)

        # set the trained model, train results, and KFold object as attributes of the object
        self.std_clf = std_clf
        self.train_results = train_results
        self.kf = kf
        
        # return the trained model, train scores, and KFold object
        return self.std_clf, self.train_results, self.kf


    def cv_predict(self):
        
        # Predict the target values on the training set using cross-validation
        y_pred_train = cross_val_predict(self.std_clf, self.X_train, self.y_train, cv=self.kf)
        
        # Calculate the performance metrics of the model
        r2 = r2_score(self.y_train, y_pred_train)
        mae = mean_absolute_error(self.y_train, y_pred_train)
        rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        
        # Store the performance metrics and the predicted target values
        cv_metrics = {'Initial_Model': self.selected_model,
              "Mean R^2": r2, 
              "Mean MAE": mae,
              "Mean RMSE": rmse,
              "CV Scores": y_pred_train}
        return cv_metrics
    ###########################################################################################################################################
    def cv_score(self):

        # Predict the target values on the training set using cross-validation
        mae_fold = cross_val_score(self.std_clf, self.X_train, self.y_train, cv=self.kf, scoring='neg_mean_absolute_error')
        mse_fold = cross_val_score(self.std_clf, self.X_train, self.y_train, cv=self.kf, scoring='neg_mean_squared_error')
        # Store the performance metrics and the predicted target values
        cv_metrics = {
            'Initial_Model': self.selected_model,
            'MAE per Fold': mae_fold,
            'Mean MAE': np.mean(mae_fold),
            '\nMSE per Fold': mse_fold,
            'Mean MSE': np.mean(mse_fold)
            }
        return cv_metrics
    
    def validate_model(self, X_test, y_test):
        y_pred = self.std_clf.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        validation_metrics = {'Initial_Model': self.selected_model,
              "Mean R^2": r2, 
              "Mean MAE": mae,
              "Mean RMSE": rmse}
        return validation_metrics

    def test_model(self, X_test, y_test):
        y_pred = self.std_clf.predict(X_test)
        return y_pred
    
    def visualize_results(self, X_test, y_test, y_pred):
        fig, ax = plt.subplots(1, 1, figsize=(32, 10))
        plt.plot(y_test)
        plt.plot(y_pred)
        plt.xlabel("Date time")
        plt.ylabel("Price Elasticity of Demand")
        plt.xticks(np.arange(0, len(y_test), 20), rotation='vertical')
        plt.show()

    def initial_model_metrics(self, X_test, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        initial_metrics = {'Initial_Model': self.selected_model,
              "Mean R^2": r2, 
              "Mean MAE": mae,
              "Mean RMSE": rmse}
        return initial_metrics

class DataInspection:
    def __init__(self, raw_data, data, expected_dtypes, processed_dtypes, inspect_cols, eng_feature_cols, target_col):
        self.raw_data = raw_data
        self.data = data
        self.expected_dtypes = expected_dtypes
        self.processed_dtypes = processed_dtypes
        self.inspect_cols = inspect_cols
        self.eng_feature_cols = eng_feature_cols
        self.target_col = target_col
        
    def inspection_1_outlier_analysis(self):                
        # Test 1: Outlier analysis
        length_raw_data = len(self.raw_data)
        length_data = len(self.data)
        perc_change = length_data / length_raw_data
        
        if length_raw_data == length_data:
            return print('Inspection Part 3 FAILURE: Outliers have not been removed')
        else:
            print(f'Inspection Part 3 PASS: Outliers removed, resultant dataset at {perc_change*100}%.')
    
        
    def inspection_2_correlation_analysis(self):    
        # Test 1: Check engineered features have adequate correlation to target
        corr_threshold = 0.1
        correlations = {}
        for col in self.eng_feature_cols:
            corr = self.data[col].corr(self.data[self.target_col])
            correlations[col] = corr
            
        below_threshold = [col for col, corr in correlations.items() if corr < corr_threshold]
        
        if below_threshold:
            print(f"Inspection Part 1 CAUTION: The following columns have a correlation below {corr_threshold} with the target variable:")
            for col in below_threshold:
                print(f"         {col}, {correlations[col]}")
        else:
            print("Inspection Part 1 PASS: All feature columns meet the correlation threshold.")
    
    def inspection_3_skewness_check(self):        
        # Test 2: Check for skewness in data
        skew_threshold = 1

        for col in self.eng_feature_cols:
            col_skew = skew(self.data[col])
            if abs(col_skew) > skew_threshold:
                print(f"Inspection Part 2 CAUTION: The {col} column is highly skewed (skewness = {col_skew:.2f}).")
            else:
                print('Inspection Part 2 PASS: All feature columns have low skew')
    

    def inspection_1_fold_bias(self):
        #Post feature engineering
        
        # Test 1: Check fold metrics for bias
        mae_threshold = 2 * np.mean(cv_metrics['MAE per Fold'])
        mse_threshold = 2 * np.mean(cv_metrics['MSE per Fold'])
        
        for i, mae in enumerate(cv_metrics['MAE per Fold']):
            if abs(mae) > mae_threshold:
                print(f"CAUTION: MAE for fold {i+1} is {mae}, above the threshold of {mae_threshold}")

        for i, mse in enumerate(cv_metrics['MSE per Fold']):
            if abs(mse) > mse_threshold:
                print(f"CAUTION: MSE for fold {i+1} is {mse}, above the threshold of {mse_threshold}")
        