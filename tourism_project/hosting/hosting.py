import os

import httpx
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

hosting_repo = "sam-vimes/tourism_prediction"

# Upload to Hugging Face
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))
# Check if the space exists, if not create it
try:
    api.repo_info(repo_id=hosting_repo, repo_type=repo_type)
    print(f"Space '{hosting_repo}' already exists. Using it.")
except (HfHubHTTPError, httpx.HTTPStatusError):
    print(f"Space '{hosting_repo}' not found. Creating new space...")
    create_repo(
        repo_id=hosting_repo, repo_type=repo_type, private=False, space_sdk="docker"
    )
    print(f"Space '{hosting_repo}' created.")

# upload the deployment folder to the HF space
api.upload_folder(
    folder_path="tourism_project/deployment",  # the local folder containing your files
    repo_id=hosting_repo,  # the target repo
    repo_type=repo_type,  # dataset, model, or space
    path_in_repo="",  # optional: subfolder path inside the repo
)
