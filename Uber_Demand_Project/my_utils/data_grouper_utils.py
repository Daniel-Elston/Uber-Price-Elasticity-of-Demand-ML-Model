class DataGrouper:
    def __init__(self, data, col_group):
        '''
        A class that groups data by a specified column

        Parameters:
        - data (pandas.DataFrame): The data to be grouped
        - col_group (str): The name of the column by which to group the data
        '''
        self.data = data
        self.col_group = col_group
    
    def group_by_hour(self, col_group):
        '''
        Group the data by hour and calculate the average passenger count and fare amount

        Parameters:
        - col_group (str): The name of the column by which to group the data

        Returns:
        - self.data (pandas.DataFrame): The grouped data
        '''
        # Group the data by hour and calculate the mean passenger count and fare amount
        self.data = self.data[col_group].groupby(['pickup_datetime'])['passenger_count','fare_amount'].agg({'passenger_count':'mean','fare_amount':'mean'}).reset_index()

        # Return the grouped data
        return self.data
    
    def group_by_day(self, col_group):
        '''
        Group the data by day and calculate the average passenger count and fare amount

        Parameters:
        - col_group (str): The name of the column by which to group the data

        Returns:
        - self.data (pandas.DataFrame): The grouped data
        '''
        # Group the data by day and calculate the mean passenger count and fare amount
        self.data = self.data[col_group].groupby(['Label_Date'])['passenger_count','fare_amount'].agg({'passenger_count':'mean','fare_amount':'mean'}).reset_index()

        # Return the grouped data
        return self.data
