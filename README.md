# diabetes_pred_app
### PART A: ENVIRONMENT SETUP
To execute the project follow the steps below: NOTE: Pycharm is used as IDE:
1. Clone the repository
2. Set up a virtual environment in the project's root directory
3. Open a terminal in the project directory and run:
   1. python3 -m virtualenv venv
   2. source venv/bin/activate
   3. pip3 install -r requirements.txt

### PART B; PROJECT EXECUTION
#### Step 1: Integrate Kaggle API to fetch the diabetes prediction dataset
1. pip install kaggle
2. Get your API key and token from your Kaggle account (This will download a file kaggle.json in your computer)
3. Ensure that your python binaries are on your path by making sure that the kaggle.json is in the user/user_name/.kaggle directory
4. Now search the desired dataset using this command: `kaggle datasets list -s [KEYWORD]`
5. Download the desired dataset using this command: `kaggle datasets download -d [DATASET]`
6. Now you will have a zip file of the dataset, unzip it and your dataset is ready to be used.


#### Step 2: Exploratory data analysis: dataset_eda.ipynb
1. Explore the data for possible issues and pre-processing steps
2. Issue found- Imbalanced datasets, number of cases without diabetes are around 92% & only 8% cases are diabetes. 
3. Used Scikit learn's SMOTE library to balance the data.
4. Checked the correlation and removed unwanted features


#### Step 3: Training & Validation: training_validation.py
1. Used Scikit-Learn's randomForest classification algorithm to detect diabetes
2. Instead of predicting 0(no-diabetes) 1(diabetes), using probability of getting diabetes is provided in the prediction. 
3. Validation results provide a precision, recall and accuracy of around 0.94. (Add visualizations)
4. Save the validated model for prediction

#### Step 4: Deploy the streamlit app: app.py
1. Make a UI using streamlit library that takes input features & the model to predict diabetes probability
2. Added result rendering along with explanation of each feature to be input

#### Step 5: Create and run a docker file: Dockerfile
1. Install docker in the system
2. Create and run a docker file in the repository to containerize the streamlit app. 
3. The Dockerfile has set of commands that will be executed sequentially to build a docker image for the streamlit application and using this image Docker will create a container for your application
4. Once the Dockerfile is made, run the command ` docker image build -t streamlit-app .`


#### Step 6: Deploy on AWS cloud
1. Launch an EC2 instance and Generate key pair and get the Public DNS address
2. Install docker in ec2 instance
3. Install git on ec2 instance
4. Clone the repo & change to the directory
5. Build the docker image of streamlit app: `sudo docker build -t streamlit-app .`
6. Run the Docker container: Start a Docker container using the built image, and map the desired port using:`docker container run -p 8501:8501 -d streamlit-app`
7. Use the following command: `sudo docker run -P streamlit-app` This will give you details of the https address where you can find your app.


#### To restart the container follow:
1. Start the EC2 instance- click on connect
2. Copy the ssh and paste it into the project terminal & change to the project directory using cd diabetes_pred_app
3. Start a Docker container using the built image, and map the desired port using:`docker container run -p 8501:8501 -d streamlit-app` (If successful this will give you an image code)
4. To get the app link run : `sudo docker run -P streamlit-app`