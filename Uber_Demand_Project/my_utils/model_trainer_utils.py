from my_utils.dependencies import *

class ModelTrainer:
    def __init__(self, data, X_train, y_train, selected_model, cv_splits):
        """
        A class used to train the selected model on the training data
        
        Parameters:
        -----------
        data: pandas DataFrame
            The entire dataset.
        X_train: pandas DataFrame
            The training input features.
        y_train: pandas DataFrame
            The training output variable.
        selected_model: sklearn estimator
            The selected machine learning algorithm.
        cv_splits: int
            The number of cross validation splits to use.
        """
        self.data = data
        self.X_train = X_train
        self.y_train = y_train
        self.selected_model = selected_model
        self.cv_splits = cv_splits
    
    def train_model(self):
        """
        Trains the selected model on the training data
        
        Returns:
        --------
        std_clf: sklearn estimator
            The trained pipeline.
        train_results: list
            A list of the scores for each fold of the training data.
        kf: KFold object
            The KFold object used to split the training data.
        """
        
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
        """
        Predict the target values on the training set using cross-validation.

        Returns:
            dict: A dictionary containing the initial model, mean R^2, mean MAE, mean RMSE, and CV scores.
        """
        
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

    
    
    def cv_score(self):
        """
        Cross-validates the selected model on the training set and calculates the mean MAE and mean MSE of the model. 
        
        Returns:
        cv_metrics: dict, a dictionary that contains the performance metrics of the model
        """
        # Predict the target values on the training set using cross-validation
        mae_fold = cross_val_score(self.std_clf, self.X_train, self.y_train, cv=self.kf, scoring='neg_mean_absolute_error')
        mse_fold = cross_val_score(self.std_clf, self.X_train, self.y_train, cv=self.kf, scoring='neg_mean_squared_error')
        
        # Store the performance metrics and the predicted target values
        cv_metrics = {
            'Initial_Model': self.selected_model,
            'MAE per Fold': mae_fold,
            'Mean MAE': np.mean(mae_fold),
            'MSE per Fold': mse_fold,
            'Mean MSE': np.mean(mse_fold)
            }
        return cv_metrics

    def validate_model(self, X_test, y_test):
        """
        Validates the selected model on the test set and calculates the mean MAE, mean RMSE and mean R^2 of the model. 
        
        Args:
        X_test: pandas DataFrame, the test set features
        y_test: pandas DataFrame, the test set target values
        
        Returns:
        validation_metrics: dict, a dictionary that contains the performance metrics of the model
        """
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
        """
        Predict target values using the trained model on the test set.
        """
        y_pred = self.std_clf.predict(X_test)
        return y_pred
        
    def visualize_results(self, X_test, y_test, y_pred):
        """
        Visualize the predicted and actual target values on the test set.
        """
        fig, ax = plt.subplots(1, 1, figsize=(32, 10))
        plt.plot(y_test)
        plt.plot(y_pred)
        plt.xlabel("Date time")
        plt.ylabel("Price Elasticity of Demand")
        plt.xticks(np.arange(0, len(y_test), 20), rotation='vertical')
        plt.show()

    def initial_model_metrics(self, X_test, y_test, y_pred):
        """
        Calculate performance metrics of the initial model on the test set.
        """
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
        initial_metrics = {'Initial_Model': self.selected_model,
            "Mean R^2": r2, 
            "Mean MAE": mae,
            "Mean RMSE": rmse}
        return initial_metrics

