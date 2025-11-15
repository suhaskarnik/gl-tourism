
from huggingface_hub import create_repo, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

data_repo = "sam-vimes/tourism_data"
repo_type = "dataset"
data_dir = "tourism_project/data"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=data_repo, repo_type=repo_type)
    print(f"Space '{data_repo}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{data_repo}' not found. Creating new space...")
    create_repo(repo_id=data_repo, repo_type=repo_type, private=False)
    print(f"Space '{data_repo}' created.")

api.upload_folder(folder_path=data_dir, repo_id=data_repo, repo_type="dataset")
