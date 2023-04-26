import pandas as pd


class CreateDataDictionary:

    def __init__(self, df):
        '''
        This class provides functions to quickly develop a data dictionary for your data set

        Parameters:
        - df (pandas.DataFrame): The data set to be documented
        '''
        self.df = df

    def make_my_data_dictionary(self, dataFrame):
        '''
        Create an initial data dictionary excluding definitions for meaning of features

        Parameters:
        - dataFrame (pandas.DataFrame): The data set to be documented

        Returns:
        - df_DD (pandas.DataFrame): A data dictionary for the input data set
        '''
        df_cols = dataFrame.columns
        df_DataDict = {}

        # Iterate over each column in the data set and add information to the dictionary
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

        # Convert the dictionary to a data frame and return it
        df_DD = pd.DataFrame(df_DataDict)
        return df_DD

    def define_data_meaning(self, df_data_dictionary):
        '''
        Quickly provide input regarding each columns meaning and transpose into a usable dictionary

        Parameters:
        - df_data_dictionary (pandas.DataFrame): The data dictionary to be updated

        Returns:
        - df_data_dictionary (pandas.DataFrame): The updated data dictionary
        '''
        df_cols = df_data_dictionary.columns
        d = 'Definition'

        # Prompt the user to provide a definition for each column in the data dictionary
        for col in df_cols:
            df_data_dictionary[col][d] = input('Provide a data definition for {}'.format(col))

        # Transpose the data dictionary and return it
        df_data_dictionary = df_data_dictionary.transpose()
        return df_data_dictionary

    def update_dd_definition(self, df_data_dictionary, attribute):
        '''
        Update the definition of a single column in the data dictionary

        Parameters:
        - df_data_dictionary (pandas.DataFrame): The data dictionary to be updated
        - attribute (str): The name of the column to be updated

        Returns:
        - df_dd (pandas.DataFrame): The updated data dictionary
        '''
        try:
            df_dd = df_data_dictionary.transpose()
            df_dd[attribute]['Definition'] = input('Provide a data definition for {}'.format(attribute))
            return df_dd
        except:
            print('Sorry, there was an error. Check attribute name and try again')

