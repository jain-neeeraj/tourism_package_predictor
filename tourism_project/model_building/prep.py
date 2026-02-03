# for data manipulation
import pandas as pd
import numpy as np # Import numpy for NaN handling
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/neeraj-jain/turism-package-prediction/tourism.csv"

try:
    bank_dataset = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Fallback for local testing if HF dataset is not accessible directly
    if os.path.exists("tourism.csv"):
        bank_dataset = pd.read_csv("tourism.csv")
        print("Loaded tourism.csv from local file.")
    else:
        raise

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'CityTier',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

# Remove 'CustomerID' column
if 'CustomerID' in bank_dataset.columns:
    bank_dataset.drop('CustomerID', axis=1, inplace=True)
    print("'CustomerID' column removed.")

# Handle missing values
print("Handling missing values...")
# Impute numerical features with mean
for col in numeric_features:
    if col in bank_dataset.columns and bank_dataset[col].isnull().any():
        bank_dataset[col].fillna(bank_dataset[col].mean(), inplace=True)
        print(f"Filled missing values in numerical column '{col}' with mean.")

# Impute categorical features with mode
for col in categorical_features:
    if col in bank_dataset.columns and bank_dataset[col].isnull().any():
        bank_dataset[col].fillna(bank_dataset[col].mode()[0], inplace=True)
        print(f"Filled missing values in categorical column '{col}' with mode.")


# Define predictor matrix (X) using selected numeric and categorical features
X = bank_dataset[numeric_features + categorical_features]

# Define target variable
y = bank_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42,   # Ensures reproducibility by setting a fixed random seed
    stratify=y         # Stratify to maintain class distribution in splits
)

print("Data split into training and testing sets.")

# Save processed data splits to CSV files
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)
print("Processed data splits saved locally as CSV files.")

files_to_upload = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
upload_repo_id = "neeraj-jain/turism-package-prediction"

# Upload files to Hugging Face Hub
print(f"Uploading files to Hugging Face Hub (repo_id: {upload_repo_id})...")
for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"data_splits/{file_path}",  # Store in a subfolder within the repo
        repo_id=upload_repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to {upload_repo_id}.")
print("All data splits uploaded to Hugging Face Hub.")
