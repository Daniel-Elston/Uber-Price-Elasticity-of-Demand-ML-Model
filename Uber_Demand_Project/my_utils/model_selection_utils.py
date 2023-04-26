from my_utils.dependencies import *

class ModelSelection:
    def __init__(self, data, target_column, cv_splits, test_split):
        """
        Initializes the ModelSelection object.

        Args:
        data (pd.DataFrame): The dataset to use for modeling.
        target_column (str): The name of the target variable column.
        cv_splits (int): The number of cross-validation splits to use.
        test_split (float): The percentage of the data to use for testing.
        """
        self.data = data
        self.target_column = target_column
        self.cv_splits = cv_splits
        self.test_split = test_split
        
        # Define a dictionary of regression models to evaluate
        self.models = {
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=0),
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(), 
            'SVR': SVR(C=1.0, epsilon=0.2), 
            'XGBRegressor': xgboost.XGBRegressor(n_estimators=100, max_depth=5, eta=0.1, subsample=1-test_split)
        }

        
    def split_dataset(self):
        """
        Splits the data into training and testing sets.
        
        Returns:
        (tuple): tuple containing:
            X_train (pd.DataFrame): Training feature data.
            X_test (pd.DataFrame): Testing feature data.
            y_train (pd.Series): Training target variable data.
            y_test (pd.Series): Testing target variable data.
        """
        input_features = self.data.drop(self.target_column, axis=1)
        target_variable = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(input_features, target_variable, shuffle=False, test_size=self.test_split)
        return X_train, X_test, y_train, y_test

    
    def model_evaluation(self, model, X_train, y_train, cv_splits):
        """
        Evaluate the performance of a given model using cross-validation.

        Args:
        - model: a regression model object that implements a fit and predict method
        - X_train: a pandas dataframe of shape (n_samples, n_features) containing the training input features
        - y_train: a pandas series of shape (n_samples,) containing the training target variable
        - cv_splits: the number of cross-validation splits to perform

        Returns:
        - a dictionary with keys 'Mean R^2', 'Mean MAE', and 'Mean RMSE' and corresponding values
        representing the average R^2 score, MAE, and RMSE across the cross-validation splits, respectively
        """
        # Create a pipeline with standardization followed by the given model
        std_clf = make_pipeline(StandardScaler(), model)

        # Create lists to store the R^2, MAE, and RMSE scores
        r2_scores = []
        mae_scores = []
        rmse_scores = []

        # Create a KFold object for splitting the training set
        kf = KFold(n_splits=cv_splits, shuffle=False)

        # Loop over the cross-validation splits and compute the R^2, MAE, and RMSE scores
        for train_index, test_index in kf.split(X_train):
            # Split the training set
            X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

            # Fit the pipeline to the training set and predict on the test set
            std_clf.fit(X_train_cv, y_train_cv)
            y_pred = std_clf.predict(X_test_cv)

            # Compute the R^2, MAE, and RMSE scores and append to the lists
            r2_scores.append(r2_score(y_test_cv, y_pred))
            mae_scores.append(mean_absolute_error(y_test_cv, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test_cv, y_pred)))

        # Compute the mean R^2, mean MAE, and mean RMSE across the cross-validation splits
        return {"Mean R^2": sum(r2_scores) / cv_splits, 
                "Mean MAE": sum(mae_scores) / cv_splits,
                "Mean RMSE": sum(rmse_scores) / cv_splits
                }

    def run_model_evaluation(self):
        """
        Splits the dataset into training and test sets and evaluates the performance of each model in the list 
        of models defined in the initialization of the class by calling the model_evaluation() method. Returns a 
        dictionary of evaluation results for each model.

        Returns:
        -------
        results: dict
            A dictionary containing the evaluation results for each model.
        """
        # Split dataset into training and test sets
        X_train, X_test, y_train, y_test = self.split_dataset()
        # Evaluate each model in the list of models using the model_evaluation() method
        results = {}
        for model_name, model in self.models.items():
            results[model_name] = self.model_evaluation(model, X_train, y_train, self.cv_splits)
        # Return a dictionary of evaluation results for each model
        return results

    
    def select_initial_model(self):
        """
        Perform grid search to identify the best hyperparameters for each model and select the model with the best score.

        Returns:
            selected_model: instance of the selected model class with the optimal hyperparameters
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = self.split_dataset()

        # Initialize variables to keep track of the best model and its score
        best_model = None
        best_score = 0

        # Loop through each model in self.models
        for model_name, model in self.models.items():
            # Define hyperparameters for grid search based on the model
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
                    
            # Perform grid search to identify the best hyperparameters for the model
            grid = GridSearchCV(estimator=model, param_grid=params, cv=self.cv_splits, n_jobs=-1)
            grid.fit(X_train, y_train)
                
            # Determine if this model is the best so far based on the grid search score
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                best_params = grid.best_params_

        # Determine the name of the best model's class and select the class
        best_model_name = type(best_model).__name__
        if best_model_name == 'XGBRegressor':
            model_class = xgboost.XGBRegressor
        else:
            model_class = getattr(sklearn.ensemble, best_model_name)

        # Instantiate the selected model class with the optimal hyperparameters
        selected_model = model_class(**best_params)
            
        # Print out the selected model's name, hyperparameters, and score
        print('\n')
        print("Selected model: {}".format(type(best_model).__name__))
        print("Hyperparameters: {}".format(best_params))
        print("Model score: {}".format(best_score))
        print('\n')

        return selected_model

