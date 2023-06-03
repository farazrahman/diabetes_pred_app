import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# clean the data before preprocessing
# TODO: break this long function later
def preprocess_data(data_str: str):
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

    # List of categorical columns to label encode
    categorical_cols = ['gender', 'hypertension', 'age_cat', 'heart_disease']
    # Perform label encoding on the categorical columns
    label_encoder = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda x: label_encoder.fit_transform(x))

    # Scale the numeric columns
    numeric_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(df.info())
    return df


def augment_data(df: pd.DataFrame):
    """
    Augment the data using smote
    :param df:
    :return:
    """
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    # Applying SMOTE to oversample the minority class
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # Create a new resampled dataset
    resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='diabetes')],
                             axis=1)
    # Check the class distribution in the balanced dataset
    print(resampled_df['diabetes'].value_counts())
    print(resampled_df['diabetes'].unique())
    return resampled_df


def _evaluation_metrics(actuals, prediction, model, X_val):
    # Evaluate the classifier
    print(classification_report(actuals, prediction))

    # plot confusion matrix
    cm = confusion_matrix(actuals, prediction, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title('Confusion matrix of validation data on base model')
    plt.show()

    # plot the feature importances
    fe = model.feature_importances_.argsort()
    plt.barh(X_val.columns[fe], model.feature_importances_[fe])
    plt.xlabel("Feature Importance")
    plt.title('Feature Importance chart')
    plt.show()


def _predict_probabilities(model, X_val, y_val):

    #calculate the probabilities of getting diabetes instead of just 0 or 1
    probabilities = model.predict_proba(X_val).tolist()

    # add probabilities column in the test data set to check the predictions & probabilities side by side
    new_val = X_val
    new_val = new_val.reset_index()
    prob_df = pd.DataFrame({'probabilities': probabilities})
    prob_df.reset_index(inplace=True)
    label_vals = y_val.to_frame()
    label_vals.reset_index(inplace=True)
    lst = [new_val, label_vals, prob_df]
    validated_df = pd.concat(lst, axis=1)
    validated_df.drop(columns=['index'], inplace=True)
    print(validated_df)


def train_model(resampled_df: pd.DataFrame):
    """
    This function takes the augmented dataframe, gets a random sample without replacement and trains a
    classification ML model and outputs the pickled model and model metrics.
    :param resampled_df: The augmented dataframe
    :return: pickled model, model metrics and graphs
    """
    # sample 1% data
    sampled_df = resampled_df.sample(frac=0.02, random_state=42)
    print(sampled_df.info())
    # Split the balanced dataset into training and testing sets
    X = sampled_df.drop('diabetes', axis=1)
    y = sampled_df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a random forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Train the classifier on the training set
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # check metrics
    _evaluation_metrics(actuals=y_test, prediction=y_pred, model=rf_classifier, X_val=X_test)

    # predict probabilities
    _predict_probabilities(model=rf_classifier, X_val=X_test,y_val=y_test)


if __name__ == '__main__':
    df = preprocess_data('diabetes_prediction_dataset.csv')
    resampled_df = augment_data(df)

    train_model(resampled_df=resampled_df)
