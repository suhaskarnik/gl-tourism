from huggingface_hub import HfApi
import os

hosting_repo = "sam-vimes/tourism_prediction"

# Upload to Hugging Face
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))
# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=hosting_repo, repo_type=repo_type)
    print(f"Space '{hosting_repo}' already exists. Using it.")
except (HfHubHTTPError, httpx.HTTPStatusError) as e:
    print(f"Space '{hosting_repo}' not found. Creating new space...")
    create_repo(repo_id=hosting_repo, repo_type=repo_type, private=False)
    print(f"Space '{hosting_repo}' created.")

api.upload_folder(
    folder_path="tourism_project/hosting",     # the local folder containing your files
    repo_id=hosting_repo,          # the target repo
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
