# diabetes_pred_app

#### Step 1: Integrate Kaggle API to fetch the diabetes prediction dataset
1. pip install kaggle
2. Get your API key and token from your Kaggle account (This will download a file kaggle.json in your computer)
3. Ensure that your python binaries are on your path by making sure that the kaggle.json is in the user/user_name/.kaggle directory
4. Now search the desired dataset using this command: `kaggle datasets list -s [KEYWORD]`
5. Download the desired dataset using this command: `kaggle datasets download -d [DATASET]`
6. Now you will have a zip file of the dataset, unzip it and your dataset is ready to be used.


#### Step 2: Exploratory data analysis
1. Explore the data for possible issues and pre-processing steps
2. Issue found- Imbalanced datasets, number of cases without diabetes are around 92% & only 8% cases are diabetes. 
3. Used Scikit learn's SMOTE library to balance the data.
4. Checked the correlation and removed unwanted features


#### Step 3: Training & Validation
1. Used Scikit-Learn's randomForest classification algorithm to detect diabetes
2. Instead of predicting 0(no-diabetes) 1(diabetes), using probability of getting diabetes is provided in the prediction. 
3. Validation results provide a precision, recall and accuracy of around 0.94. (Add visualizations)
4. Save the validated model for prediction

#### Step 4: Deploy the streamlit app
1. Make a UI using streamlit library that takes input features & the model to predict diabetes probability
2. Added result rendering along with explanation of each feature to be input


#### Step 5: Create and run a docker file
1. Install docker in the system
2. Create and run a docker file in the repository to containerize the streamlit app. 
3. The Dockerfile has set of commands that will be executed sequentially to build a docker image for the streamlit application and using this image Docker will create a container for your application
4. Once the Dockerfile is made, run the command ` docker build -t streamlit-app .`
5. Check the newly built image `docker image ls`
6. Once the image is built you can run the app in a container using the command; `docker run -p 8501:8501 streamlit-app` or `docker run -P streamlit-app`
7. Check the containerized app in `http://localhost:8501/`
`