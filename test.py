import pandas as pd
import requests

# Load dataset
dataset_path = "training_dataset.csv"
df = pd.read_csv(dataset_path)

# API endpoint
API_URL = "http://localhost:8000/api/question"

# Test each question in the dataset
for index, row in df.iterrows():
    question = row["question"]
    response = requests.post(API_URL, data={"question": question})
    
    if response.status_code == 200:
        print(f"Question: {question}\nAnswer: {response.json().get('answer')}\n")
    else:
        print(f"Question: {question}\nError: {response.status_code}, {response.text}\n")
