import pandas as pd


# clean the data before preprocessing
def clean_data(data_str: str):
    """
    This function takes in the dataset name str, loads in the dataframe
    format, clear out all duplicate rows and unwanted columns
    :param data_str: dataset name string
    :return: cleaned df
    """
    df = pd.read_csv(data_str)
    # drop duplicate rows
    df.drop_duplicates(ignore_index=True, inplace=True)
    # drop the other category in gender
    df = df[df['gender'] != 'Other']
    # drop smoking history column as it is not clear in the metadata
    df.drop(columns='smoking_history', inplace=True)
    # Feature Engineering
    # make age category instead of age factor
    df['age_cat'] = pd.cut(df['age'], bins=[0, 5, 12, 18, 30, 40, 50, 60, 120],
                           labels=['0-5', '6-12', '13-18', '19-30', '30-40', '40-50', '50-60', '>60'])
    df.drop(columns='age', inplace=True)
    print(df.info())
    return df


if __name__ == '__main__':
    clean_data('diabetes_prediction_dataset.csv')
