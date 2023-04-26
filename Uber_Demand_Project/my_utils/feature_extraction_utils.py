class FeatureExtraction:
    def __init__(self, data, col_feature, n_window):
        """
        Initialize FeatureExtraction object.

        Parameters:
            data (pandas DataFrame): Input dataset.
            col_feature (list of str): Column names of features to be extracted.
            n_window (int): Window size for rolling average.

        """
        self.data = data
        self.col_feature = col_feature
        self.n_window = n_window

    def calc_pct_change(self, col_feature):
        """
        Calculate the percentage change for each feature column.

        Parameters:
            col_feature (list of str): Column names of features to be extracted.

        Returns:
            self.data (pandas DataFrame): DataFrame with percentage change columns added.

        """
        for col in col_feature:
            self.data[col + '_pct_change'] = self.data[col].pct_change()
        return self.data
    
    def calc_sma(self, col, n_window):
        """
        Calculate the simple moving average for a given feature column.

        Parameters:
            col (str): Column name of feature to be extracted.
            n_window (int): Window size for rolling average.

        Returns:
            self.data (pandas DataFrame): DataFrame with simple moving average column added.

        """
        self.data[col + '_sma'] = self.data[col].rolling(window=self.n_window).mean()
        return self.data
        
    def calc_ema(self, col, n_window):
        """
        Calculate the exponential moving average for a given feature column.

        Parameters:
            col (str): Column name of feature to be extracted.
            n_window (int): Window size for rolling average.

        Returns:
            self.data (pandas DataFrame): DataFrame with exponential moving average column added.

        """
        self.data[col + '_ema'] = self.data[col].ewm(span=n_window, adjust=False).mean()
        return self.data
        
    def calc_sma_PED(self):
        """
        Calculate the ratio of passenger count to fare amount using simple moving average.

        Returns:
            self.data (pandas DataFrame): DataFrame with sPED column added.

        """
        self.data['sPED'] = self.data.passenger_count_pct_change_sma / self.data.fare_amount_pct_change_sma
        return self.data
        
    def calc_ema_PED(self):
        """
        Calculate the ratio of passenger count to fare amount using exponential moving average.

        Returns:
            self.data (pandas DataFrame): DataFrame with ePED column added.

        """
        self.data['ePED'] = self.data.passenger_count_pct_change_ema / self.data.fare_amount_pct_change_ema
        return self.data

