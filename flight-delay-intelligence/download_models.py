import os
from azure.storage.blob import BlobServiceClient

AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
ACCOUNT_NAME = "flightdelaymodels"
CONTAINER = "models"
MODELS_DIR = "/app/models"

FILES = [
    "lgbm_delay_classifier_final.pkl",
    "lgbm_delay_regressor_final.pkl",
    "model_final_metadata.json",
    "feature_list_final.txt",
    "route_risk_scores.parquet",
    "shap_values_50k_v2.npy",
    "shap_sample_X_v2.parquet",
    "shap_sample_y_v2.parquet",
    "dashboard_stats.json",
    "dashboard_monthly.parquet",
    "dashboard_hourly.parquet",
    "dashboard_carrier.parquet",
    "airline_rankings.parquet",
    "cascade_tail_lookup.parquet",
    "test_flights_slim.parquet",
]

def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    client = BlobServiceClient(
        account_url=f"https://{ACCOUNT_NAME}.blob.core.windows.net",
        credential=AZURE_STORAGE_KEY
    )
    container = client.get_container_client(CONTAINER)

    for fname in FILES:
        dest = os.path.join(MODELS_DIR, fname)
        if os.path.exists(dest):
            print(f"  Already exists: {fname}")
            continue
        print(f"  Downloading {fname}...")
        with open(dest, "wb") as f:
            f.write(container.download_blob(fname).readall())
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  ✓ {fname} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    print("Downloading models from Azure Blob Storage...")
    download_models()
    print("All files ready.")
