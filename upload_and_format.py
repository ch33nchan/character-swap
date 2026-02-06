#!/usr/bin/env python3
"""
Upload images to Azure Blob Storage using SAS token and format CSV with =IMAGE() formulas
Requires: pip install azure-storage-blob pandas pillow
Set env vars:
  export AZURE_STORAGE_SAS_TOKEN="your_sas_token"
  export CDN_BASE_URL="https://dev-content.dashtoon.ai"  # Optional, uses blob URL if not set
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except ImportError:
    print("ERROR: azure-storage-blob not installed")
    print("Run: pip install azure-storage-blob")
    sys.exit(1)

# Configuration - from your working upload.py
STORAGE_ACCOUNT = "dashprodstore"  # Production storage account
CONTAINER_NAME = "stability-images"  # Production container
CDN_BASE_URL = "https://content.dashtoon.ai"
BLOB_PREFIX = "face-swap-results"  # Prefix for organized storage

# Production SAS token (expires 2030-06-30)
AZURE_SAS_TOKEN_PROD = "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2030-06-30T21:36:20Z&st=2024-06-30T13:36:20Z&spr=https,http&sig=ewQCKuZEeC7A6vnlFxSDxDwVU7zunyCwB4tfE6880HA%3D"


def get_blob_client():
    """Get Azure Blob Service Client using SAS token"""
    account_url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    return BlobServiceClient(account_url=f"{account_url}?{AZURE_SAS_TOKEN_PROD}")


def upload_image_to_azure(image_path: Path, blob_name: str) -> str:
    """Upload image to Azure Blob Storage and return CDN URL"""
    try:
        blob_service = get_blob_client()
        container_client = blob_service.get_container_client(CONTAINER_NAME)
        
        # Upload blob
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(image_path, "rb") as data:
            blob_client.upload_blob(
                data, 
                overwrite=True,
                content_settings=ContentSettings(content_type="image/png")
            )
        
        # Get CDN URL
        url = f"{CDN_BASE_URL}/{CONTAINER_NAME}/{blob_name}"
        
        print(f"✓")
        return url
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return ""


def to_image_formula(url: str) -> str:
    """Convert URL to Google Sheets =IMAGE() formula"""
    if url and url.startswith("http"):
        return f'=IMAGE("{url}")'
    return url


def main():
    parser = argparse.ArgumentParser(description="Upload face swap results to Azure and add =IMAGE() columns")
    parser.add_argument("--csv", required=True, help="Input CSV path (with Swap_Status etc.)")
    parser.add_argument("--output-csv", help="Output CSV path (default: input_with_azure_urls.csv)")
    args = parser.parse_args()
    csv_path = args.csv
    output_dir = Path("data/output")

    if not Path(csv_path).exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return
    
    print("="*60)
    print("Azure Blob Storage Upload & CSV Formatter")
    print("="*60)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(df)} rows")
    
    # Add new columns
    if 'Swap_Azure_URL' not in df.columns:
        df['Swap_Azure_URL'] = ''
    if 'Swap_Image_Formula' not in df.columns:
        df['Swap_Image_Formula'] = ''
    
    # Convert existing URL columns to =IMAGE() formulas
    url_columns = ['Original Image', 'Swapped Image', 'Generated Image', 'Reference Angle', 'Front Angle']
    print(f"\nConverting {len(url_columns)} URL columns to =IMAGE() formulas...")
    
    for col in url_columns:
        if col in df.columns:
            formula_col = f'{col}_Formula'
            if formula_col not in df.columns:
                df[formula_col] = ''
            
            for idx in range(len(df)):
                url = df.at[idx, col]
                if pd.notna(url) and str(url).startswith('http'):
                    df.at[idx, formula_col] = to_image_formula(str(url))
    
    print(f"✓ Converted {len(url_columns)} columns to formulas")
    
    # Upload result images from data/output
    if not output_dir.exists():
        print(f"\nWARNING: {output_dir} not found, skipping image uploads")
    else:
        print(f"\nUploading images from {output_dir}...")
        
        # Get list
        row_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
        total = len(row_dirs)
        print(f"Found {total} result directories")
        
        success_count = 0
        
        for i, row_dir in enumerate(row_dirs, 1):
            try:
                row_num = int(row_dir.name.split('_')[1])
                idx = row_num - 1
                
                result_path = row_dir / "result.png"
                if result_path.exists():
                    print(f"[{i}/{total}] Row {row_num}...", end=" ", flush=True)
                    
                    # Upload with organized blob name
                    blob_name = f"{BLOB_PREFIX}/row_{row_num:04d}/result.png"
                    azure_url = upload_image_to_azure(result_path, blob_name)
                    
                    if azure_url:
                        df.at[idx, 'Swap_Azure_URL'] = azure_url
                        df.at[idx, 'Swap_Image_Formula'] = to_image_formula(azure_url)
                        success_count += 1
                        
            except Exception as e:
                print(f"\n✗ Error: {e}")
        
        print(f"\n✓ Uploaded {success_count}/{total} images")
    
    # Save updated CSV
    output_csv = args.output_csv or csv_path.replace('.csv', '_with_azure_urls.csv')
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Output CSV: {output_csv}")
    print(f"\nNew columns added:")
    print(f"  - Swap_Azure_URL: Direct Azure Blob URL")
    print(f"  - Swap_Image_Formula: =IMAGE() formula for result")
    print(f"  - *_Formula: =IMAGE() formulas for all URL columns")
    print(f"\nTotal images uploaded: {success_count if output_dir.exists() else 0}")


if __name__ == "__main__":
    main()
