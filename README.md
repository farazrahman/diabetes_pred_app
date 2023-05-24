# diabetes_pred_app

#### Step 1: Integrate Kaggle API to fetch the diabetes prediction dataset
1. pip install kaggle
2. Get your API key and token from your Kaggle account (This will download a file kaggle.json in your computer)
3. Ensure that your python binaries are on your path by making sure that the kaggle.json is in the user/user_name/.kaggle directory
4. Now search the desired dataset using this command: `kaggle datasets list -s [KEYWORD]`
5. Download the desired dataset using this command: `kaggle datasets download -d [DATASET]`
6. Now you will have a zip file of the dataset, unzip it and your dataset is ready to be used.