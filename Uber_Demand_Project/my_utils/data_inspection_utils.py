class DataInspection:
    def __init__(self, raw_data, data, expected_dtypes, processed_dtypes, inspect_cols, eng_feature_cols, target_col):
        '''Initialize the DataInspection object with the necessary attributes'''
        self.raw_data = raw_data # Original raw data
        self.data = data # Processed data
        self.expected_dtypes = expected_dtypes # Expected datatypes for the data columns
        self.processed_dtypes = processed_dtypes # Datatypes of the processed data columns
        self.inspect_cols = inspect_cols # Columns to inspect for outliers and skewness
        self.eng_feature_cols = eng_feature_cols # Engineered feature columns
        self.target_col = target_col # Target variable column
        
    def inspection_1_outlier_analysis(self):                
        '''Perform the outlier analysis check'''
        # Test 1: Outlier analysis
        length_raw_data = len(self.raw_data)
        length_data = len(self.data)
        perc_change = length_data / length_raw_data
        
        if length_raw_data == length_data:
            return print('Inspection Part 3 FAILURE: Outliers have not been removed')
        else:
            print(f'Inspection Part 3 PASS: Outliers removed, resultant dataset at {perc_change*100}%.')
    
        
    def inspection_2_correlation_analysis(self):    
        '''Perform the correlation analysis check'''
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
        '''Perform the skewness check'''
        # Test 2: Check for skewness in data
        skew_threshold = 1

        for col in self.eng_feature_cols:
            col_skew = skew(self.data[col])
            if abs(col_skew) > skew_threshold:
                print(f"Inspection Part 2 CAUTION: The {col} column is highly skewed (skewness = {col_skew:.2f}).")
            else:
                print('Inspection Part 2 PASS: All feature columns have low skew')
    

    def inspection_1_fold_bias(self, cv_metrics):
        '''Perform the cross-validation fold bias check'''
        # Test 1: Check fold metrics for bias
        mae_threshold = 2 * np.mean(cv_metrics['MAE per Fold'])
        mse_threshold = 2 * np.mean(cv_metrics['MSE per Fold'])
        
        for i, mae in enumerate(cv_metrics['MAE per Fold']):
            if abs(mae) > mae_threshold:
                print(f"CAUTION: MAE for fold {i+1} is {mae}, above the threshold of {mae_threshold}")

        for i, mse in enumerate(cv_metrics['MSE per Fold']):
            if abs(mse) > mse_threshold:
                print(f"CAUTION: MSE for fold {i+1} is {mse}, above the threshold of {mse_threshold}")

        