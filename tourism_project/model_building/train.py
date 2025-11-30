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
from sklearn.metrics import f1_score, precision_score, recall_score

# for model training, tuning, and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

api = HfApi()
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
            "Designation",
        ],
    )
)

model = RandomForestClassifier(random_state=42, class_weight="balanced")
pipe = make_pipeline(preprocessor, model)

# When training locally, an MLFlow UI can be spawned based on the below DB.
#   That is NOT possible when we use GitHub Actions (GHA), because GHA does not have persistent storage
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("MLOps_Tourism")

param_grid = {
    "randomforestclassifier__max_depth": [6, 7, 8, None],
    "randomforestclassifier__max_leaf_nodes": [80, 90, 100, None],
    "randomforestclassifier__min_samples_split": [2, 8, 16],
    "randomforestclassifier__min_samples_leaf": [1, 2, 3, 4],
}


with mlflow.start_run():
    # Grid Search

    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_f1", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Evaluation
    mlflow.log_metrics(
        {
            "train_F1": f1_score(y_train, y_pred_train),
            "train_Recall": recall_score(y_train, y_pred_train),
            "train_Precision": precision_score(y_train, y_pred_train),
            "test_F1": f1_score(y_test, y_pred_test),
            "test_Recall": recall_score(y_test, y_pred_test),
            "test_Precision": precision_score(y_test, y_pred_test),
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".joblib"
    ) as temp_joblib_file:
        temp_file_path = temp_joblib_file.name

        # Save the model locally
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
            path_or_fileobj=temp_file_path,
            path_in_repo="best_tourism_model_v1.joblib",
            repo_id=model_repo,
            repo_type=repo_type,
        )
