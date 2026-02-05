#!/usr/bin/env python3
"""
Update CSV with results from existing output files
Run this if the main script crashed before saving the CSV
"""

import json
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

def get_image_metrics(image_path: Path) -> dict:
    try:
        img = Image.open(image_path)
        file_size = image_path.stat().st_size
        width, height = img.size
        return {
            'width': width,
            'height': height,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
        }
    except:
        return {}

def main():
    csv_path = "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv"
    output_dir = Path("data/output")
    
    df = pd.read_csv(csv_path)
    print(f"CSV has {len(df)} rows")
    
    # Add columns if not present
    for col in ['Swap_Status', 'Swap_Output_Path', 'Swap_Time_Sec', 
                'Swap_Output_Size_MB', 'Swap_Output_Width', 'Swap_Output_Height',
                'Swap_Face_Detected', 'Swap_Face_Confidence']:
        if col not in df.columns:
            df[col] = ''
    
    # Scan output directories
    success_count = 0
    for row_dir in sorted(output_dir.iterdir()):
        if not row_dir.is_dir():
            continue
        
        try:
            row_num = int(row_dir.name.split('_')[1])
            idx = row_num - 1
            
            result_path = row_dir / "result.png"
            if result_path.exists():
                metrics = get_image_metrics(result_path)
                
                df.at[idx, 'Swap_Status'] = 'success'
                df.at[idx, 'Swap_Output_Path'] = str(result_path)
                df.at[idx, 'Swap_Output_Size_MB'] = metrics.get('file_size_mb', 0)
                df.at[idx, 'Swap_Output_Width'] = metrics.get('width', 0)
                df.at[idx, 'Swap_Output_Height'] = metrics.get('height', 0)
                success_count += 1
                print(f"Row {row_num}: success ({metrics.get('width', 0)}x{metrics.get('height', 0)}, {metrics.get('file_size_mb', 0)}MB)")
        except Exception as e:
            print(f"Error processing {row_dir}: {e}")
    
    # Mark missing rows as failed
    for idx in range(len(df)):
        if df.at[idx, 'Swap_Status'] == '':
            df.at[idx, 'Swap_Status'] = 'not_processed'
    
    # Save updated CSV
    output_csv = csv_path.replace('.csv', '_results.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nUpdated CSV saved to: {output_csv}")
    print(f"Total successful: {success_count}/{len(df)}")
    
    # Also create a simple results.json
    results_data = {
        'summary': {
            'total_rows': len(df),
            'successful': success_count,
            'failed': len(df) - success_count,
            'success_rate': round(success_count / len(df) * 100, 1)
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"Results summary saved to: results.json")

if __name__ == "__main__":
    main()
