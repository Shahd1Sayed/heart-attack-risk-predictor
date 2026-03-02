import os
from huggingface_hub import HfApi, login
import getpass

def deploy():
    print("🚀 Preparing to deploy to Hugging Face Spaces...")
    
    # Check if user is logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Logged in as: {user_info['name']}")
    except:
        print("\n🔑 You are not logged in to Hugging Face.")
        print("Please get an Access Token with WRITE permissions from: https://huggingface.co/settings/tokens")
        token = getpass.getpass("Enter your HF Access Token: ")
        login(token)
        api = HfApi()

    repo_id = "Shahd1sayed/heart-attack-risk-predictor"
    
    print(f"\n📦 Uploading files to {repo_id}...")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            allow_patterns=[
                "app.py",
                "Dockerfile",
                "model.pkl",
                "requirements.txt",
                "static/**",
                "README.md"
            ],
            commit_message="🚀 Deploying production web app via HF API (fixes binary git tracking)"
        )
        print("\n🎉 Deployment successful!")
        print(f"🌍 Your app will be live at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("Please make sure the Space exists and your token has WRITE access.")

if __name__ == "__main__":
    deploy()
