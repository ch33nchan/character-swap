#!/usr/bin/env python3
"""
Upload all face swap results to Azure and update CSV with IMAGE() formulas.
Run locally: ensure tools.lighting_transfer is on PYTHONPATH or use --tools-path.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image
from loguru import logger

upload_image_to_storage = None


def load_uploader(tools_path: str = None):
    global upload_image_to_storage
    if tools_path:
        path = Path(tools_path).resolve()
        if path.is_dir():
            sys.path.insert(0, str(path))
        else:
            sys.path.insert(0, str(path.parent))
    try:
        from tools.lighting_transfer import upload_image_to_storage as _upload
        upload_image_to_storage = _upload
        return True
    except ImportError:
        pass
    for parent in [ROOT.parent, ROOT.parent.parent]:
        if (parent / "tools" / "lighting_transfer.py").exists():
            sys.path.insert(0, str(parent))
            try:
                from tools.lighting_transfer import upload_image_to_storage as _upload
                upload_image_to_storage = _upload
                return True
            except ImportError:
                pass
    return False


def upload_image(image_path: Path, prefix: str = "face-swap") -> str:
    """Upload image to Azure and return URL"""
    try:
        img = Image.open(image_path).convert("RGB")
        url = upload_image_to_storage(img, prefix=prefix)
        logger.success(f"Uploaded: {image_path.name} -> {url}")
        return url
    except Exception as e:
        logger.error(f"Failed to upload {image_path}: {e}")
        return ""


def to_image_formula(url: str) -> str:
    """Convert URL to Google Sheets IMAGE formula"""
    if url and url.startswith("http"):
        return f'=IMAGE("{url}")'
    return url


def main():
    parser = argparse.ArgumentParser(description="Upload face swap results to Azure and add IMAGE() formulas to CSV")
    parser.add_argument("--tools-path", type=str, help="Path to repo containing tools/lighting_transfer.py (for local run)")
    parser.add_argument("--csv", type=str, default="Master - Tech Solutioning - Char Const - Rerun with head_eye gaze_results.csv", help="Input CSV path")
    args = parser.parse_args()
    
    if not load_uploader(args.tools_path):
        logger.error("Could not import upload_image_to_storage. Run with --tools-path /path/to/repo that contains tools/lighting_transfer.py")
        sys.exit(1)
    
    csv_path = args.csv
    output_dir = Path("data/output")
    
    if not Path(csv_path).exists():
        logger.error(f"CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Add new columns for Azure URLs and IMAGE formulas
    if 'Swap_Azure_URL' not in df.columns:
        df['Swap_Azure_URL'] = ''
    if 'Swap_Image_Formula' not in df.columns:
        df['Swap_Image_Formula'] = ''
    
    # Convert existing URL columns to IMAGE formulas
    url_columns = ['Original Image', 'Swapped Image', 'Generated Image', 'Reference Angle', 'Front Angle']
    for col in url_columns:
        formula_col = f'{col}_Formula'
        if col in df.columns:
            if formula_col not in df.columns:
                df[formula_col] = ''
            for idx in range(len(df)):
                url = df.at[idx, col]
                if pd.notna(url) and str(url).startswith('http'):
                    df.at[idx, formula_col] = to_image_formula(str(url))
    
    # Upload result images and update CSV
    success_count = 0
    for row_dir in sorted(output_dir.iterdir()):
        if not row_dir.is_dir():
            continue
        
        try:
            row_num = int(row_dir.name.split('_')[1])
            idx = row_num - 1
            
            result_path = row_dir / "result.png"
            if result_path.exists():
                logger.info(f"Processing row {row_num}...")
                
                # Upload to Azure
                azure_url = upload_image(result_path, prefix=f"face-swap/row_{row_num:04d}")
                
                if azure_url:
                    df.at[idx, 'Swap_Azure_URL'] = azure_url
                    df.at[idx, 'Swap_Image_Formula'] = to_image_formula(azure_url)
                    success_count += 1
                    
        except Exception as e:
            logger.error(f"Error processing {row_dir}: {e}")
    
    # Save updated CSV
    output_csv = csv_path.replace('.csv', '_with_urls.csv')
    df.to_csv(output_csv, index=False)
    logger.info(f"\nSaved updated CSV to: {output_csv}")
    logger.info(f"Uploaded {success_count} images to Azure")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total rows: {len(df)}")
    print(f"Images uploaded: {success_count}")
    print(f"Output CSV: {output_csv}")
    print("\nNew columns added:")
    print("  - Swap_Azure_URL: Direct Azure CDN URL")
    print("  - Swap_Image_Formula: =IMAGE() formula for Google Sheets")
    print("  - *_Formula columns: =IMAGE() formulas for existing URL columns")


if __name__ == "__main__":
    main()
