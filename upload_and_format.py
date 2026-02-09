#!/usr/bin/env python3
"""
Upload result images to Azure Blob and write minimal CSV:
Swapped Image, Reference Angle, Front Angle, new image
Requires: pip install azure-storage-blob pandas
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except ImportError:
    print("ERROR: azure-storage-blob not installed. Run: pip install azure-storage-blob")
    sys.exit(1)

STORAGE_ACCOUNT = "dashprodstore"
CONTAINER_NAME = "stability-images"
CDN_BASE_URL = "https://content.dashtoon.ai"
BLOB_PREFIX = "face-swap-results"
AZURE_SAS_TOKEN_PROD = "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2030-06-30T21:36:20Z&st=2024-06-30T13:36:20Z&spr=https,http&sig=ewQCKuZEeC7A6vnlFxSDxDwVU7zunyCwB4tfE6880HA%3D"


def get_blob_client():
    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=f"{account_url}?{AZURE_SAS_TOKEN_PROD}")


def upload_image_to_azure(image_path: Path, blob_name: str) -> str:
    try:
        blob_service = get_blob_client()
        container_client = blob_service.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        with open(image_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="image/png"))
        return f"{CDN_BASE_URL}/{CONTAINER_NAME}/{blob_name}"
    except Exception as e:
        print(f" Error: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Upload results to Azure; output minimal CSV (Swapped Image, Reference Angle, Front Angle, new image)")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--output-csv", help="Output CSV path (default: input_with_new_image.csv)")
    args = parser.parse_args()
    csv_path = args.csv
    output_dir = Path("data/output")

    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "Swapped Image" not in df.columns:
        print("ERROR: Missing required column: Swapped Image")
        return
    if "Reference Angle" not in df.columns:
        df["Reference Angle"] = ""
    if "Front Angle" not in df.columns:
        df["Front Angle"] = ""
    out_df = df[["Swapped Image", "Reference Angle", "Front Angle"]].copy()
    out_df["new image"] = ""

    upload_details = []
    success_count = 0

    if output_dir.exists():
        row_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
        allowed_rows = set(range(1, len(out_df) + 1))
        filtered_row_dirs = []
        for row_dir in row_dirs:
            try:
                row_num = int(row_dir.name.split("_")[1])
            except Exception:
                continue
            if row_num in allowed_rows:
                filtered_row_dirs.append(row_dir)
        total = len(filtered_row_dirs)
        print(f"Uploading {total} images from {output_dir} for CSV rows 1..{len(out_df)}...")
        for i, row_dir in enumerate(filtered_row_dirs, 1):
            try:
                row_num = int(row_dir.name.split("_")[1])
                idx = row_num - 1
                result_path = row_dir / "result.png"
                if result_path.exists():
                    print(f"[{i}/{total}] Row {row_num}...", end=" ", flush=True)
                    blob_name = f"{BLOB_PREFIX}/row_{row_num:04d}/result.png"
                    azure_url = upload_image_to_azure(result_path, blob_name)
                    if azure_url:
                        out_df.at[idx, "new image"] = azure_url
                        success_count += 1
                        upload_details.append({"row": row_num, "path": str(result_path), "blob_name": blob_name, "azure_url": azure_url})
            except Exception as e:
                print(f"\nError: {e}")
        print(f"\nUploaded {success_count}/{total} images")
    else:
        print(f"WARNING: {output_dir} not found, skipping uploads")

    output_csv = args.output_csv or csv_path.replace(".csv", "_with_new_image.csv")
    out_df.to_csv(output_csv, index=False)

    results_json_path = Path(csv_path).resolve().parent / "upload_results.json"
    with open(results_json_path, "w") as f:
        json.dump({"source_csv": csv_path, "output_csv": output_csv, "total_rows": len(df), "uploaded_count": success_count, "details": upload_details}, f, indent=2)

    print(f"Minimal CSV: {output_csv}")
    print(f"Upload details: {results_json_path}")


if __name__ == "__main__":
    main()
