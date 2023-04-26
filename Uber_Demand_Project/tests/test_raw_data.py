from my_utils.dependencies import *
from my_utils.file_loading_utils import file_load
from my_utils.data_processing_utils import DatasetProcessor
@pytest.fixture
def raw_data():
    path = 'C:/Users/delst/OneDrive - Queen Mary, University of London/Desktop/VSCode/Advanced_Projects/Uber_Demand_Project/data_archive/*.csv'
    input_files = glob.glob(path)
    data = file_load(input_files)[0]
    return data

@pytest.fixture
def data_dict():
    path = 'C:/Users/delst/OneDrive - Queen Mary, University of London/Desktop/VSCode/Advanced_Projects/Uber_Demand_Project/Data_dict_raw.csv'
    input_files = glob.glob(path)
    df_dd = file_load(input_files)[0]
    return df_dd


def test_empty_df(raw_data):
    assert not raw_data.empty, 'Dataframe is empty'
    assert raw_data is not None, 'Dataframe is None'
    # assert not raw_data.empty, 'Dataframe is empty'
    
def test_missing_values(raw_data):
    assert not raw_data.isnull().values.any(), 'Dataframe contains missing values'

def test_nan_values(raw_data):
    # assert not raw_data.isna().values.any(), 'Dataframe contains NaNs'
    nans = raw_data.isna().sum()
    assert nans == 0, f"Found {nans} NaN values"
    
def test_for_duplicate_rows(raw_data):
    duplicates = raw_data[raw_data.duplicated()]
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate rows"

# def test_data_types(data_dict):
#     # Define the expected data types for each column in the DataFrame
#     raw_dtypes = data_dict.Type.to_dict().values()
#     expected_dtypes = data_dict.Expected_Type.to_dict().values()
#     assert raw_dtypes == expected_dtypes, 'Unexpected dtype detected'

def test_data_types(data_dict, raw_data):
    # Define the expected data types for each column in the DataFrame
    expected_dtypes = data_dict.Expected_Type.to_dict().values()
    assert tuple(raw_data.dtypes) == tuple(expected_dtypes), 'Unexpected dtype detected'