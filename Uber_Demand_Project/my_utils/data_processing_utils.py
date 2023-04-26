from my_utils.dependencies import pd, np, stats

class DatasetProcessor:
    def __init__(self, data, threshold):
        """
        DatasetProcessor constructor.

        Parameters:
        - data (pandas.DataFrame): the dataset to be processed.
        - threshold (float): the threshold for winsorization.
        """
        self.data = data
        self.threshold = threshold
    
    def data_transform(self):
        """
        Transforms the dataset by extracting relevant features from pickup_datetime.

        Returns:
        - self.data (pandas.DataFrame): the transformed dataset.
        """
        # Convert pickup_datetime to datetime format and remove timezone information
        self.data['pickup_datetime'] = pd.to_datetime(self.data['pickup_datetime']).dt.tz_localize(None)
        # Round pickup_datetime down to the hour
        self.data['pickup_datetime'] = self.data['pickup_datetime'].dt.floor('H')
        
        # Extract hour and date from pickup_datetime
        self.data['Label_Hour'] = self.data['pickup_datetime'].dt.hour
        self.data['Label_Date'] = self.data['pickup_datetime'].dt.date

        # Extract day name and day of the week from pickup_datetime
        self.data['day_name'] = pd.to_datetime(self.data['pickup_datetime']).dt.day_name()
        self.data['day_of_the_week'] = pd.to_datetime(self.data['pickup_datetime']).dt.weekday
        return self.data
            
    def data_remove_outliers(self):
        """
        Removes outliers from the dataset using z-scores.

        Returns:
        - self.data (pandas.DataFrame): the dataset with outliers removed.
        """
        # Remove rows where passenger_count z-score or fare_amount z-score is greater than 2
        self.data = self.data[(np.abs(stats.zscore(self.data['passenger_count'])) < 2)]
        self.data = self.data[(np.abs(stats.zscore(self.data['fare_amount'])) < 2)]
        return self.data

    def data_clean(self):
        """
        Cleans the dataset by removing rows with null values and replacing 0 values in
        passenger_count and fare_amount with 1.

        Returns:
        - self.data (pandas.DataFrame): the cleaned dataset.
        """
        # Replace 0 values in passenger_count and fare_amount with 1
        self.data.passenger_count.replace(0, 1, inplace=True)
        self.data.fare_amount.replace(0, 1, inplace=True)
        # Drop rows with null values
        self.data = self.data.dropna()
        # Take absolute value of fare_amount
        self.data['fare_amount'] = abs(self.data.fare_amount)
        return self.data
    
    def winsorization(self):
        """
        Applies winsorization to the dataset by setting values outside the threshold range to the
        median value.

        Returns:
        - self.data (pandas.DataFrame): the dataset with winsorization applied.
        """
        # Calculate median, lower quantile, and upper quantile for sPED
        median = self.data['sPED'].median()
        lower_quantile = self.data['sPED'].quantile(self.threshold)
        upper_quantile = self.data['sPED'].quantile(1-self.threshold)
        # Set values outside the threshold range to the median value
        self.data['sPED'][self.data['sPED'] < lower_quantile] = -1*median
        self.data['sPED'][self.data['sPED'] > upper_quantile] = median
        return self.data
    
    def index_set(self):
        """
        Sets the index of the dataset to pickup_datetime in the format YYYY-MM-DD-HH.
        """
        self.data = self.data.set_index('pickup_datetime')
        self.data.index = self.data.index.strftime('%Y-%m-%d-%H')
        return self.data

