import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# Load the model
clf_model = joblib.load("rfc_model.pkl")
feature_names = clf_model.feature_names_in_
print(feature_names)

# prepare the input data according to the processing of training data

def predict_diabetes(gender, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, age_cat):
    if gender == 'F':
        gender = '0'
    else:
        gender = '1'

    if hypertension == 'Yes':
        hypertension = '1'
    else:
        hypertension = '0'

    if age_cat == '0-5':
        age_cat = 0
    elif age_cat == '6-12':
        age_cat = 1
    elif age_cat == '13-18':
        age_cat = 2
    elif age_cat == '19-30':
        age_cat = 3
    elif age_cat == '30-40':
        age_cat = 4
    elif age_cat == '40-50':
        age_cat = 5
    elif age_cat == '50-60':
        age_cat = 6
    elif age_cat == '>60':
        age_cat = 7

    if heart_disease == 'No':
        heart_disease = 0
    elif heart_disease == 'Yes':
        heart_disease = 1

    # Perform the prediction
    test_df = pd.DataFrame([[gender, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, age_cat]],
                           columns=['gender', 'hypertension', 'heart_disease', 'bmi',
                                    'HbA1c_level', 'blood_glucose_level', 'age_cat'])
    predict_output = clf_model.predict(test_df)
    probability = clf_model.predict_proba(test_df).tolist()

    fe = clf_model.feature_importances_.argsort()
    fig, ax = plt.subplots()
    ax.barh(test_df.columns[fe], clf_model.feature_importances_[fe])
    ax.set_xlabel("Feature Importance")
    ax.set_title('Feature Importance chart')

    return predict_output, probability, fig


st.title('Diabetes Prediction')

st.header('Enter the mandatory fields')

gender = st.selectbox('gender:', ['F', 'M'])
hypertension = st.selectbox('hypertension:', ['Yes', 'No'])
age_cat = st.selectbox('age_cat:', ['0-5', '6-12', '13-18', '19-30', '30-40', '40-50', '50-60', '>60'])
heart_disease = st.selectbox('heart_disease:', ['No', 'Yes'])
bmi = st.number_input('bmi:', min_value=0, max_value=40, value=1)
HbA1c_level = st.number_input('HbA1c_level:', min_value=0, max_value=15, value=1)
blood_glucose_level = st.number_input('blood_glucose_level:', min_value=0, max_value=1000, value=1)

if st.button('Predict Diabetes'):
    hd, proba, fig = predict_diabetes(gender, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, age_cat)
    # print(hd, proba, fig)
    # st.success(f'The overall prediction is {hd[0]}')
    st.success(f'You have around {round(proba[0][1], 2) * 100} % chances of being diabetic')
    st.success(f'Please consult your doctor before taking any medication')

    st.header('Legend')
    col1, col2 = st.columns(2)
    col1.metric("No diabetes", "0")
    col2.metric("Yes diabetes", "1")

    # st.header('Factors affecting the prediction in decreasing order')
    # st.pyplot(fig=fig)

    st.header('For more information check below')

    st.text("Gender: gender of the patient [M: Male, F: Female]")
    st.text("Hypertension: Do you have hypertension?")
    st.text("Heart Disease: Do you have heart disease?")
    st.text("BMI: what is your BMI [formula: weight (kg) / [height (m)]2]")
    st.text("HbA1c_level: A hemoglobin A1C (HbA1C) test is a blood test that shows what your average blood sugar "
            "(glucose) level was over the past two to three months")
    st.text("blood_glucose_level: Fasting blood sugar level")
