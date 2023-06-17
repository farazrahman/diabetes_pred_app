import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
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


st.title(':blue[Welcome to Diabytes]')
st.subheader(':blue[The Diabetes Prediction App!!!]')
image = Image.open('glucose-meter.png')
st.image(image, width=250)

st.header('Enter the mandatory fields')

gender = st.selectbox('Select your :red[Gender] F-Female, M-Male', ['F', 'M'])
hypertension = st.selectbox('Do you have :red[hypertension?]', ['Yes', 'No'])
age_cat = st.selectbox('What is your :red[age category?]', ['0-5', '6-12', '13-18', '19-30', '30-40', '40-50', '50-60', '>60'])
heart_disease = st.selectbox('Do you have :red[heart disease?]', ['No', 'Yes'])
bmi = st.slider('Select you :red[BMI(Weight in Kg/Height in meters)]', min_value=0, max_value=60, value=1)
HbA1c_level = st.slider('Select your :red[HbA1c_level] (Normal: <5.7%, Pre-diabetes: 5.7-6.4%, Type 2 diabetes: >6.5%)', min_value=0, max_value=10, value=1)
blood_glucose_level = st.slider('Select your :red[fasting blood_glucose_level] (Normal: <99 mg/dL, Pre-diabetes: 100-125 mg/dL, Diabetic: >126 mg/dL)', min_value=0, max_value=500, value=1)

if st.button('Predict Diabetes'):
    hd, proba, fig = predict_diabetes(gender, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level,
                                      age_cat)
    # print(hd, proba, fig)
    # st.success(f'The overall prediction is {hd[0]}')
    st.success(f'You have around {round(proba[0][1], 2) * 100} % chances of being diabetic')
    st.warning(f'Please consult your doctor before taking any medication', icon="🚨")

    # st.header('Legend')
    # col1, col2 = st.columns(2)
    # col1.metric("No diabetes", "0")
    # col2.metric("Yes diabetes", "1")
    st.divider()
    st.subheader('Factors affecting the prediction in decreasing order')
    image2 = Image.open('fe_img.png')
    st.image(image2)
    st.divider()
    st.subheader(':blue[For more information check below]')

    st.text("Gender: gender of the patient [M: Male, F: Female]")
    st.text("Hypertension: Do you have hypertension?")
    st.text("Heart Disease: Do you have heart disease?")
    st.text("BMI: what is your BMI [formula: weight (kg) / [height (m)]2]")
    st.text("HbA1c_level: A hemoglobin A1C (HbA1C %) test is a blood test that shows what your average blood sugar "
            "(glucose) level was over the past two to three months")
    st.text("blood_glucose_level: Fasting blood sugar level(mg/dL)")
