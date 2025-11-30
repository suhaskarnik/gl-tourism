import os
import tempfile

import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

api = HfApi(token=os.getenv("HF_TOKEN"))

data_repo = "sam-vimes/tourism_data"
DATASET_PATH = f"hf://datasets/{data_repo}/tourism.csv"

df = pd.read_csv(DATASET_PATH, index_col=0)

# Cleaning the gender column
df.loc[df["Gender"] == "Fe Male", "Gender"] = "Female"


print("Dataset loaded successfully.")

target = "ProdTaken"
X = df.drop(columns=[target, "CustomerID"])

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

print(
    f"""Shapes
    {X_train.shape=}\t{y_train.shape=},
    {X_test.shape=}\t{y_test.shape=},
"""
)

Xs = [X_train, X_test]
ys = [y_train, y_test]
labels = ["train", "test"]

datasets = [X_train, X_test, y_train, y_test]
files = []
for dataset in datasets:
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv"
    ) as temp_csv_file:
        temp_file_path = temp_csv_file.name

        dataset.to_csv(temp_file_path, index=False)
        files.append(temp_file_path)

for file_path, name in zip(files, ["X_train", "X_test", "y_train", "y_test"]):
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="prepped/" + name + ".csv",  # just the filename
        repo_id=data_repo,
        repo_type="dataset",
    )
