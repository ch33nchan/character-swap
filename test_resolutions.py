#!/usr/bin/env python3
"""
Test face swap at multiple resolutions: 1K, 2K, 4K
Measures timing for each resolution
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from PIL import Image

DEFAULT_COMFYUI_URL = "http://localhost:8189"
DOWNLOAD_TIMEOUT = 30
COMFYUI_TIMEOUT = 600

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def download_image(url: str, output_path: Path) -> bool:
    try:
        logger.info(f"Downloading {url}")
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def upload_to_comfyui(server_url: str, image_path: Path) -> Optional[str]:
    try:
        url = f"{server_url}/upload/image"
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/png')}
            data = {'overwrite': 'true'}
            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()
        return response.json().get('name', image_path.name)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return None


def modify_workflow_resolution(workflow: Dict, generated_img: str, original_img: str, 
                                output_prefix: str, megapixels: float) -> Dict:
    """Modify workflow with images and resolution"""
    workflow = json.loads(json.dumps(workflow))
    
    if "151" in workflow:
        workflow["151"]["inputs"]["image"] = generated_img
    if "121" in workflow:
        workflow["121"]["inputs"]["image"] = original_img
    if "9" in workflow:
        workflow["9"]["inputs"]["filename_prefix"] = output_prefix
    if "135" in workflow:
        workflow["135"]["inputs"]["value"] = megapixels
        logger.info(f"Resolution: {megapixels}MP")
    
    return workflow


def queue_workflow(server_url: str, workflow: Dict, client_id: str) -> Optional[str]:
    try:
        url = f"{server_url}/prompt"
        response = requests.post(url, json={"prompt": workflow, "client_id": client_id}, timeout=30)
        response.raise_for_status()
        prompt_id = response.json().get('prompt_id')
        logger.info(f"Queued: {prompt_id}")
        return prompt_id
    except Exception as e:
        logger.error(f"Queue failed: {e}")
        return None


def wait_for_completion(server_url: str, prompt_id: str, timeout: int = COMFYUI_TIMEOUT) -> bool:
    url = f"{server_url}/history/{prompt_id}"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=10)
            history = response.json()
            
            if prompt_id in history:
                status = history[prompt_id].get('status', {})
                if status.get('completed'):
                    return status.get('status_str') == 'success'
            time.sleep(3)
        except:
            time.sleep(3)
    
    return False


def get_output_images(server_url: str, prompt_id: str):
    try:
        response = requests.get(f"{server_url}/history/{prompt_id}", timeout=10)
        history = response.json()
        
        if prompt_id in history:
            outputs = history[prompt_id].get('outputs', {})
            if "9" in outputs and 'images' in outputs["9"]:
                return outputs["9"]['images']
        return []
    except:
        return []


def download_output(server_url: str, image_info: Dict, output_path: Path) -> bool:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{server_url}/view"
        params = {
            'filename': image_info['filename'],
            'type': image_info.get('type', 'output'),
            'subfolder': image_info.get('subfolder', '')
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Get metrics
        img = Image.open(output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {output_path} ({img.size[0]}x{img.size[1]}, {size_mb:.2f}MB)")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def test_resolution(row_data: pd.Series, workflow: Dict, server_url: str, 
                    resolution_name: str, megapixels: float, output_dir: Path) -> Dict:
    """Test a single resolution"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {resolution_name} ({megapixels}MP)")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    result = {
        'resolution': resolution_name,
        'megapixels': megapixels,
        'success': False,
        'time_sec': 0,
        'output_path': '',
        'output_width': 0,
        'output_height': 0,
        'output_size_mb': 0,
        'error': None
    }
    
    try:
        # Download
        gen_url = row_data.get('Generated Image', '')
        orig_url = row_data.get('Original Image', '')
        
        gen_path = Path(f"test_input/generated.png")
        orig_path = Path(f"test_input/original.png")
        
        if not download_image(gen_url, gen_path) or not download_image(orig_url, orig_path):
            result['error'] = "Download failed"
            return result
        
        # Upload
        gen_name = upload_to_comfyui(server_url, gen_path)
        orig_name = upload_to_comfyui(server_url, orig_path)
        
        if not gen_name or not orig_name:
            result['error'] = "Upload failed"
            return result
        
        # Modify workflow
        modified = modify_workflow_resolution(
            workflow, gen_name, orig_name, 
            f"test_{resolution_name}", megapixels
        )
        
        # Execute
        process_start = time.time()
        prompt_id = queue_workflow(server_url, modified, str(int(time.time() * 1000)))
        
        if not prompt_id or not wait_for_completion(server_url, prompt_id):
            result['error'] = "Execution failed"
            return result
        
        # Download result
        images = get_output_images(server_url, prompt_id)
        if not images:
            result['error'] = "No output"
            return result
        
        result_path = output_dir / f"{resolution_name}.png"
        if not download_output(server_url, images[0], result_path):
            result['error'] = "Download output failed"
            return result
        
        # Get metrics
        img = Image.open(result_path)
        result['output_path'] = str(result_path)
        result['output_width'] = img.size[0]
        result['output_height'] = img.size[1]
        result['output_size_mb'] = round(result_path.stat().st_size / (1024 * 1024), 2)
        result['time_sec'] = round(time.time() - start_time, 2)
        result['processing_time_sec'] = round(time.time() - process_start, 2)
        result['success'] = True
        
        logger.info(f"✓ SUCCESS in {result['time_sec']}s")
        
    except Exception as e:
        result['error'] = str(e)
        result['time_sec'] = round(time.time() - start_time, 2)
        logger.error(f"Error: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Test face swap at multiple resolutions')
    parser.add_argument('--csv', required=True, help='CSV file path')
    parser.add_argument('--workflow', required=True, help='API format workflow JSON')
    parser.add_argument('--row', type=int, default=1, help='Row number to test (1-indexed)')
    parser.add_argument('--comfyui-url', default=DEFAULT_COMFYUI_URL, help='ComfyUI server URL')
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    row_data = df.iloc[args.row - 1]
    
    with open(args.workflow) as f:
        workflow_template = json.load(f)
    
    output_dir = Path(f"resolution_test_row{args.row}")
    output_dir.mkdir(exist_ok=True)
    
    # Test resolutions: 1K, 2K, 4K
    resolutions = [
        ("1K", 1.0),
        ("2K", 4.0),
        ("4K", 8.3)
    ]
    
    results = []
    total_start = time.time()
    
    for res_name, megapixels in resolutions:
        result = test_resolution(row_data, workflow_template, args.comfyui_url, 
                                res_name, megapixels, output_dir)
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n{'='*60}")
    print("RESOLUTION TEST RESULTS")
    print(f"{'='*60}")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"{status} {r['resolution']:3s}: {r['time_sec']:6.1f}s  |  {r['output_width']}x{r['output_height']}  |  {r['output_size_mb']:.2f}MB")
    
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Output directory: {output_dir}/")
    
    # Save JSON
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'row': args.row,
            'total_time_sec': round(total_time, 2),
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {output_dir}/results.json")


if __name__ == "__main__":
    main()
