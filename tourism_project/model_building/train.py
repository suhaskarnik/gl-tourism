import argparse
import os
import tempfile

import httpx
import joblib
import mlflow
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# for model training, tuning, and evaluation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# The same .py file will be used for local training, and for training in GitHub Actions.
##  If the --local argument is passed to the .py, it will run in "local" mode, and log training data to localhost:5000
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local", action="store_true")
args = parser.parse_args()
local = args.local


data_repo = "sam-vimes/tourism_data"
model_repo = "sam-vimes/tourism_model"


def get_data(dataset_name):
    local_path = hf_hub_download(
        repo_id=data_repo, filename=f"prepped/{dataset_name}.csv", repo_type="dataset"
    )
    return pd.read_csv(local_path)


X_train = get_data("X_train")
X_test = get_data("X_test")
y_train = np.ravel(get_data("y_train"))
y_test = np.ravel(get_data("y_test"))

print(
    f"""Shapes
    {X_train.shape=}\t{y_train.shape=},
    {X_test.shape=}\t{y_test.shape=},
"""
)

preprocessor = make_column_transformer(
    (
        OneHotEncoder(handle_unknown="ignore"),
        [
            "TypeofContact",
            "Occupation",
            "Gender",
            "ProductPitched",
            "MaritalStatus",
        ],
    ),
    (
        OrdinalEncoder(
            categories=[["Executive", "VP", "AVP", "Senior Manager", "Manager"]],
            # handle_unknown="use_encoded_value",
        ),
        ["Designation"],
    ),
    remainder="passthrough",
)

model = RandomForestClassifier(random_state=42, class_weight="balanced")
pipe = make_pipeline(preprocessor, model)

# When training locally, an MLFlow UI can be spawned based on the below DB.
#   That is NOT possible when we use GitHub Actions (GHA), because GHA does not have persistent storage
if local:
    # note: this cannot be used in GHA because it expects a process to listen on port 5000, and there is no such listener in the GHA environment
    mlflow.set_tracking_uri("http://localhost:5000")
else:
    # note: this SQLite DB is lost the moment the GHA runner completes.
    # In a Production workflow, this should be an external location like an S3 Bucket, RDS etc
    mlflow.set_tracking_uri("sqlite:///mlflow.db")


mlflow.set_experiment("MLOps_Tourism")

grid = {
    # number of trees
    "randomforestclassifier__n_estimators": [200, 400, 600, 800],
    # limit depth to prevent overfitting
    "randomforestclassifier__max_depth": [None, 5, 8, 12, 15],
    # minimum samples to split a node
    "randomforestclassifier__min_samples_split": [2, 5, 10, 20, 40, 80],
    # minimum samples in a leaf
    "randomforestclassifier__min_samples_leaf": [1, 2, 4, 8, 12],
    # stronger regularization than min_samples_leaf
    "randomforestclassifier__min_weight_fraction_leaf": [0.0, 0.005, 0.01, 0.02],
    # shrink feature set per split to reduce correlation between trees
    "randomforestclassifier__max_features": ["sqrt", "log2", 0.25, 0.5, 0.75],
    # extra pruning
    "randomforestclassifier__min_impurity_decrease": [0.0, 0.0001, 0.001, 0.01],
    # for imbalanced data (important!)
    "randomforestclassifier__class_weight": ["balanced", "balanced_subsample"],
}


mlflow.sklearn.autolog()

metric = "recall"
with mlflow.start_run():
    randomized_cv = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        n_iter=200,
        scoring=metric,
        cv=5,
        random_state=42,
        n_jobs=-1,
    )

    # Log parameter sets
    randomized_cv.fit(X_train, y_train)
    results = randomized_cv.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric(f"mean_{metric}", mean_score)

    # Best model
    mlflow.log_params(randomized_cv.best_params_)
    best_model = randomized_cv.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Evaluation
    mlflow.log_metrics(
        {
            "train_F1": f1_score(y_train, y_pred_train),
            "train_Recall": recall_score(y_train, y_pred_train),
            "train_Precision": precision_score(y_train, y_pred_train),
            "train_PRC": average_precision_score(y_train, y_pred_train),
            "train_ROC": roc_auc_score(y_train, y_pred_train),
            "test_F1": f1_score(y_test, y_pred_test),
            "test_Recall": recall_score(y_test, y_pred_test),
            "test_Precision": precision_score(y_test, y_pred_test),
            "test_PRC": average_precision_score(y_test, y_pred_test),
            "test_ROC": roc_auc_score(y_test, y_pred_test),
        }
    )

    if not local:  # not local = we are running in GitHub Actions, so push the model to Hugging Face
        api = HfApi()

        # create a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".joblib"
        ) as temp_joblib_file:
            # ... and get its path
            temp_file_path = temp_joblib_file.name

            # Save the model locally at that temp file path
            joblib.dump(best_model, temp_file_path)

            # Log the model artifact
            mlflow.log_artifact(temp_file_path, artifact_path="model")
            print(f"Model saved as artifact at: {temp_file_path}")

            # Upload to Hugging Face
            repo_type = "model"

            api = HfApi(token=os.getenv("HF_TOKEN"))
            try:
                api.repo_info(repo_id=model_repo, repo_type=repo_type)
                print(f"Space '{model_repo}' already exists. Using it.")
            except (HfHubHTTPError, httpx.HTTPStatusError):
                print(f"Space '{model_repo}' not found. Creating new space...")
                create_repo(repo_id=model_repo, repo_type=repo_type, private=False)
                print(f"Space '{model_repo}' created.")

            api.upload_file(
                path_or_fileobj=temp_file_path,  # temporary file from earlier
                path_in_repo="best_tourism_model_v1.joblib",  # the name in the model repo
                repo_id=model_repo,
                repo_type=repo_type,
            )
