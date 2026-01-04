import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



# Configuration

# Input data (ONLY the stratified subset)
DATA_CSV_PATH = "data/processed/train_with_images.csv"

# Where images will be saved
IMAGE_DIR = "data/images"

# Mapbox API settings
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
MAPBOX_STYLE = "satellite-v9"
ZOOM_LEVEL = 18
IMAGE_SIZE = "256x256"

# Safety controls
REQUEST_DELAY = 0.1      # seconds between requests
MAX_IMAGES = 6000        # hard safety cap

# Validation
if MAPBOX_TOKEN is None:
    raise RuntimeError(
        "MAPBOX_TOKEN not found. "
        "Please set it in your environment or .env file."
    )


# Utility Functions
def build_mapbox_url(lat: float, lon: float) -> str:
    """
    Build a Mapbox Static Images API URL for given latitude and longitude.
    """
    base_url = "https://api.mapbox.com/styles/v1/mapbox"
    return (
        f"{base_url}/{MAPBOX_STYLE}/static/"
        f"{lon},{lat},{ZOOM_LEVEL}/{IMAGE_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )


def download_image(url: str, save_path: str) -> bool:
    """
    Download an image from the given URL and save it to disk.
    Returns True if successful, False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True

        print(f"Failed request with status code {response.status_code}")
        return False

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return False

# Main Pipeline
def main():
    print("Starting satellite image fetching pipeline...")
    print(f"Reading data from: {DATA_CSV_PATH}")
    print(f"Images will be saved to: {IMAGE_DIR}")

    # Create image directory safely
    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

   
    # Load & validate data
    df = pd.read_csv(DATA_CSV_PATH)
    print(f"Total rows in image subset: {len(df)}")

    required_columns = {"id", "lat", "long"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[["id", "lat", "long"]].copy()

    if df[["lat", "long"]].isnull().any().any():
        raise ValueError("Found missing latitude or longitude values")

    # Deduplicate properties (one image per property)
    before = len(df)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    after = len(df)

    print(f"Removed {before - after} duplicate property IDs")


    if df["id"].duplicated().any():
        raise ValueError("Duplicate property IDs found")

    print("Latitude range:", df["lat"].min(), "to", df["lat"].max())
    print("Longitude range:", df["long"].min(), "to", df["long"].max())
    print("Data validation complete.")

    # Download images
    print("Starting image download...")
    downloaded = 0

    for _, row in df.iterrows():

        if downloaded >= MAX_IMAGES:
            print("Reached MAX_IMAGES limit. Stopping.")
            break

        property_id = row["id"]
        image_path = os.path.join(IMAGE_DIR, f"{property_id}.png")

        # Idempotency: skip if image already exists
        if os.path.exists(image_path):
            continue

        image_url = build_mapbox_url(
            lat=row["lat"],
            lon=row["long"]
        )

        success = download_image(image_url, image_path)

        if success:
            downloaded += 1
        else:
            print(f"Failed to download image for property ID {property_id}")

        time.sleep(REQUEST_DELAY)

    print(f"Image download completed. Total images downloaded: {downloaded}")


if __name__ == "__main__":
    main()
