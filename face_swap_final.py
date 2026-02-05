#!/usr/bin/env python3
"""
Face Swap Automation - Final Working Version
Works with ComfyUI API format workflows
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

# Configuration
DEFAULT_COMFYUI_URL = "http://localhost:8189"
MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 30
COMFYUI_TIMEOUT = 300
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('face_swap.log')
    ]
)
logger = logging.getLogger(__name__)


def download_image(url: str, output_path: Path, retries: int = MAX_RETRIES) -> bool:
    """Download image from URL with retries"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{retries})")
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                
    return False


def upload_to_comfyui(server_url: str, image_path: Path) -> Optional[str]:
    """Upload image to ComfyUI server"""
    try:
        url = f"{server_url}/upload/image"
        
        with open(image_path, 'rb') as f:
            files = {'image': (image_path.name, f, 'image/png')}
            data = {'overwrite': 'true'}
            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()
            
        result = response.json()
        return result.get('name', image_path.name)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return None


def modify_api_workflow(api_workflow: Dict, generated_img: str, original_img: str, output_prefix: str) -> Dict:
    """
    Modify API format workflow with row-specific images
    Node 151: Generated image (main input with mask)
    Node 121: Original image (reference)
    Node 9: SaveImage output
    """
    workflow = json.loads(json.dumps(api_workflow))
    
    # Update LoadImage nodes
    if "151" in workflow:
        workflow["151"]["inputs"]["image"] = generated_img
        logger.info(f"Node 151: {generated_img}")
    
    if "121" in workflow:
        workflow["121"]["inputs"]["image"] = original_img  
        logger.info(f"Node 121: {original_img}")
    
    # Update SaveImage output prefix
    if "9" in workflow:
        workflow["9"]["inputs"]["filename_prefix"] = output_prefix
        logger.info(f"Node 9: {output_prefix}")
    
    return workflow


def queue_workflow(server_url: str, workflow: Dict, client_id: str) -> Optional[str]:
    """Submit workflow to ComfyUI"""
    try:
        url = f"{server_url}/prompt"
        payload = {
            "prompt": workflow,
            "client_id": client_id
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        prompt_id = result.get('prompt_id')
        logger.info(f"Queued: {prompt_id}")
        return prompt_id
    except Exception as e:
        logger.error(f"Queue failed: {e}")
        logger.error(f"Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
        return None


def wait_for_completion(server_url: str, prompt_id: str, timeout: int = COMFYUI_TIMEOUT) -> bool:
    """Wait for workflow completion"""
    url = f"{server_url}/history/{prompt_id}"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            history = response.json()
            
            if prompt_id in history:
                status = history[prompt_id].get('status', {})
                if status.get('completed'):
                    if status.get('status_str') == 'success':
                        logger.info("Completed successfully")
                        return True
                    else:
                        logger.error(f"Failed: {status.get('status_str')}")
                        return False
            
            time.sleep(3)
        except Exception as e:
            logger.warning(f"Check failed: {e}")
            time.sleep(3)
    
    logger.error(f"Timeout after {timeout}s")
    return False


def get_output_images(server_url: str, prompt_id: str):
    """Get output images from completed workflow"""
    try:
        url = f"{server_url}/history/{prompt_id}"
        response = requests.get(url, timeout=10)
        history = response.json()
        
        images = []
        if prompt_id in history:
            for node_output in history[prompt_id].get('outputs', {}).values():
                if 'images' in node_output:
                    images.extend(node_output['images'])
        
        return images
    except:
        return []


def download_output(server_url: str, image_info: Dict, output_path: Path) -> bool:
    """Download output image from ComfyUI"""
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
        
        logger.info(f"Saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def process_row(row_data: pd.Series, row_num: int, workflow_template: Dict, server_url: str) -> bool:
    """Process single CSV row"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Row {row_num}")
    logger.info(f"{'='*60}")
    
    row_dir = f"row_{row_num:04d}"
    input_dir = Path(INPUT_DIR) / row_dir
    output_dir = Path(OUTPUT_DIR) / row_dir
    
    try:
        # Get URLs
        generated_url = row_data.get('Generated Image', '')
        original_url = row_data.get('Original Image', '')
        
        if not generated_url or not original_url:
            logger.error("Missing image URLs")
            return False
        
        # Download
        logger.info("Downloading...")
        gen_path = input_dir / "generated.png"
        orig_path = input_dir / "original.png"
        
        if not download_image(generated_url, gen_path):
            return False
        if not download_image(original_url, orig_path):
            return False
        
        # Upload
        logger.info("Uploading...")
        gen_name = upload_to_comfyui(server_url, gen_path)
        orig_name = upload_to_comfyui(server_url, orig_path)
        
        if not gen_name or not orig_name:
            return False
        
        # Modify workflow
        logger.info("Preparing workflow...")
        modified = modify_api_workflow(
            workflow_template,
            generated_img=gen_name,
            original_img=orig_name,
            output_prefix=f"{row_dir}_result"
        )
        
        # Execute
        logger.info("Executing face swap...")
        client_id = str(int(time.time() * 1000))
        prompt_id = queue_workflow(server_url, modified, client_id)
        
        if not prompt_id:
            return False
        
        if not wait_for_completion(server_url, prompt_id):
            return False
        
        # Download results
        logger.info("Downloading results...")
        images = get_output_images(server_url, prompt_id)
        if not images:
            logger.error("No output images")
            return False
        
        result_path = output_dir / "result.png"
        if not download_output(server_url, images[0], result_path):
            return False
        
        logger.info(f"✓ SUCCESS: {result_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Face swap automation using ComfyUI API')
    parser.add_argument('--csv', required=True, help='CSV file path')
    parser.add_argument('--workflow', required=True, help='API format workflow JSON')
    parser.add_argument('--comfyui-url', default=DEFAULT_COMFYUI_URL, help='ComfyUI server URL')
    parser.add_argument('--start-row', type=int, default=1, help='Start row (1-indexed)')
    parser.add_argument('--end-row', type=int, help='End row (inclusive)')
    parser.add_argument('--gpu-ids', type=str, help='GPU IDs (e.g., 0,1,2,3)')
    
    args = parser.parse_args()
    
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        logger.info(f"Using GPUs: {args.gpu_ids}")
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"CSV: {len(df)} rows")
    
    # Load workflow
    with open(args.workflow) as f:
        workflow_template = json.load(f)
    logger.info(f"Workflow: {args.workflow}")
    logger.info(f"Nodes in workflow: {len(workflow_template)}")
    
    # Process rows
    end = args.end_row if args.end_row else len(df)
    results = []
    
    for idx in range(args.start_row - 1, min(end, len(df))):
        row_num = idx + 1
        success = process_row(df.iloc[idx], row_num, workflow_template, args.comfyui_url)
        results.append({'row': row_num, 'success': success, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {successful}/{len(results)} successful")
    logger.info(f"{'='*60}")
    
    # Save results
    with open("results.json", 'w') as f:
        json.dump({'summary': {'total': len(results), 'successful': successful, 'failed': len(results) - successful}, 'results': results}, f, indent=2)
    
    logger.info(f"Results saved to: results.json")


if __name__ == "__main__":
    main()
