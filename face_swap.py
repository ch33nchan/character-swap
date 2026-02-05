#!/usr/bin/env python3
"""
Character Face Swap Automation Pipeline
Automated batch processing of face swaps using ComfyUI Flux2 Klein 9b workflow
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
import subprocess

import pandas as pd
import requests
from websocket import WebSocket

# Configuration
DEFAULT_COMFYUI_URL = "http://localhost:8188"
DEFAULT_BATCH_SIZE = 10
MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 30
COMFYUI_TIMEOUT = 300
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('face_swap.log')
    ]
)
logger = logging.getLogger(__name__)


class ImageDownloader:
    """Download images from URLs with retry logic"""
    
    @staticmethod
    def download(url: str, output_path: Path, retries: int = MAX_RETRIES) -> bool:
        """Download image from URL to output path with retries"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(retries):
            try:
                logger.info(f"Downloading {url} (attempt {attempt + 1}/{retries})")
                response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded to {output_path}")
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    
        logger.error(f"Failed to download {url} after {retries} attempts")
        return False


class ComfyUIClient:
    """Client for interacting with ComfyUI API"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.client_id = str(uuid.uuid4())
        
    def upload_image(self, image_path: Path) -> Optional[Dict]:
        """Upload image to ComfyUI server"""
        try:
            url = f"{self.server_url}/upload/image"
            
            with open(image_path, 'rb') as f:
                files = {'image': (image_path.name, f, 'image/png')}
                data = {'overwrite': 'true'}
                response = requests.post(url, files=files, data=data, timeout=30)
                response.raise_for_status()
                
            result = response.json()
            logger.info(f"Uploaded {image_path.name} to ComfyUI")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload {image_path}: {e}")
            return None
    
    def queue_prompt(self, workflow: Dict) -> Optional[str]:
        """Submit workflow to ComfyUI queue"""
        try:
            url = f"{self.server_url}/prompt"
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            prompt_id = result.get('prompt_id')
            logger.info(f"Queued workflow with prompt_id: {prompt_id}")
            return prompt_id
            
        except Exception as e:
            logger.error(f"Failed to queue workflow: {e}")
            return None
    
    def wait_for_completion(self, prompt_id: str, timeout: int = COMFYUI_TIMEOUT) -> bool:
        """Wait for workflow execution to complete via polling"""
        try:
            url = f"{self.server_url}/history/{prompt_id}"
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                history = response.json()
                
                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('completed', False):
                        logger.info(f"Workflow {prompt_id} completed successfully")
                        return True
                    elif status.get('status_str') == 'error':
                        logger.error(f"Workflow {prompt_id} failed with error")
                        return False
                
                time.sleep(2)
            
            logger.error(f"Workflow {prompt_id} timed out after {timeout}s")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for workflow completion: {e}")
            return False
    
    def get_output_images(self, prompt_id: str) -> List[Dict]:
        """Get output images from completed workflow"""
        try:
            url = f"{self.server_url}/history/{prompt_id}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            history = response.json()
            
            if prompt_id not in history:
                return []
            
            outputs = history[prompt_id].get('outputs', {})
            images = []
            
            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    images.extend(node_output['images'])
            
            return images
            
        except Exception as e:
            logger.error(f"Failed to get output images: {e}")
            return []
    
    def download_output_image(self, image_info: Dict, output_path: Path) -> bool:
        """Download output image from ComfyUI"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            filename = image_info['filename']
            subfolder = image_info.get('subfolder', '')
            type_param = image_info.get('type', 'output')
            
            url = f"{self.server_url}/view"
            params = {
                'filename': filename,
                'type': type_param,
                'subfolder': subfolder
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded output image to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download output image: {e}")
            return False


class WorkflowModifier:
    """Modify ComfyUI workflow JSON dynamically"""
    
    @staticmethod
    def modify_workflow(
        workflow: Dict,
        generated_image: str,
        original_image: str,
        reference_image: Optional[str],
        front_image: Optional[str],
        prompt: str,
        output_prefix: str
    ) -> Dict:
        """
        Modify workflow with row-specific parameters
        
        Node mapping from Flux2 Klein 9b Face Swap.json:
        - Node 151: Generated image (main input)
        - Node 121: Original image (reference 2)
        - Node 128: Reference angle (optional, currently disabled mode=4)
        - Node 137: Front angle (optional, currently disabled mode=4)
        - Node 107: CLIP text encode (prompt)
        - Node 9: Save image output
        """
        workflow = workflow.copy()
        
        # Node 151: Generated Image (main input)
        for node in workflow.get('nodes', []):
            if node['id'] == 151 and node['type'] == 'LoadImage':
                node['widgets_values'] = [generated_image, "image"]
                logger.debug(f"Set node 151 to {generated_image}")
            
            # Node 121: Original Image
            elif node['id'] == 121 and node['type'] == 'LoadImage':
                node['widgets_values'] = [original_image, "image"]
                logger.debug(f"Set node 121 to {original_image}")
            
            # Node 128: Reference Angle (optional)
            elif node['id'] == 128 and node['type'] == 'LoadImage' and reference_image:
                node['widgets_values'] = [reference_image, "image"]
                node['mode'] = 0  # Enable node if image provided
                logger.debug(f"Set node 128 to {reference_image}")
            
            # Node 137: Front Angle (optional)
            elif node['id'] == 137 and node['type'] == 'LoadImage' and front_image:
                node['widgets_values'] = [front_image, "image"]
                node['mode'] = 0  # Enable node if image provided
                logger.debug(f"Set node 137 to {front_image}")
            
            # Node 107: CLIP Text Encode (prompt)
            elif node['id'] == 107 and node['type'] == 'CLIPTextEncode':
                node['widgets_values'] = [prompt]
                logger.debug(f"Set node 107 prompt (length: {len(prompt)} chars)")
            
            # Node 9: Save Image output
            elif node['id'] == 9 and node['type'] == 'SaveImage':
                node['widgets_values'] = [output_prefix]
                logger.debug(f"Set node 9 output prefix to {output_prefix}")
        
        return workflow


class GitManager:
    """Handle git operations"""
    
    @staticmethod
    def git_command(args: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Execute git command"""
        try:
            result = subprocess.run(
                ['git'] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                logger.error(f"Git command failed: {result.stderr}")
                return False, result.stderr
                
        except Exception as e:
            logger.error(f"Git command error: {e}")
            return False, str(e)
    
    @staticmethod
    def commit_and_push(message: str, files: Optional[List[Path]] = None) -> bool:
        """Commit and push changes to git"""
        try:
            if files:
                for file_path in files:
                    success, output = GitManager.git_command(['add', str(file_path)])
                    if not success:
                        logger.warning(f"Failed to add {file_path}")
            else:
                GitManager.git_command(['add', 'data/output/'])
            
            success, output = GitManager.git_command(['commit', '-m', message])
            if not success:
                logger.warning("Git commit failed (possibly nothing to commit)")
                return False
            
            success, output = GitManager.git_command(['push'])
            if success:
                logger.info(f"Successfully pushed: {message}")
                return True
            else:
                logger.error("Git push failed")
                return False
                
        except Exception as e:
            logger.error(f"Git operation failed: {e}")
            return False


def process_row(
    row_data: pd.Series,
    row_num: int,
    workflow_template: Dict,
    comfyui_client: ComfyUIClient,
    input_base_dir: Path,
    output_base_dir: Path
) -> Tuple[bool, Optional[str]]:
    """
    Process a single CSV row
    
    Returns: (success: bool, error_message: Optional[str])
    """
    row_dir = f"row_{row_num:04d}"
    input_dir = input_base_dir / row_dir
    output_dir = output_base_dir / row_dir
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing Row {row_num}")
    logger.info(f"{'='*60}")
    
    try:
        # Extract URLs from CSV
        original_url = row_data.get('Original Image', '')
        generated_url = row_data.get('Generated Image', '')
        reference_url = row_data.get('Reference Angle', '')
        front_url = row_data.get('Front Angle', '')
        prompt = row_data.get('Edit Prompt', '')
        
        if not original_url or not generated_url:
            return False, "Missing required image URLs"
        
        if not prompt:
            return False, "Missing edit prompt"
        
        # Download images
        logger.info("Step 1: Downloading images...")
        
        generated_path = input_dir / "generated.png"
        if not ImageDownloader.download(generated_url, generated_path):
            return False, "Failed to download generated image"
        
        original_path = input_dir / "original.png"
        if not ImageDownloader.download(original_url, original_path):
            return False, "Failed to download original image"
        
        reference_path = None
        if reference_url and reference_url.strip():
            reference_path = input_dir / "reference.png"
            if not ImageDownloader.download(reference_url, reference_path):
                logger.warning("Failed to download reference image, continuing...")
                reference_path = None
        
        front_path = None
        if front_url and front_url.strip():
            front_path = input_dir / "front.png"
            if not ImageDownloader.download(front_url, front_path):
                logger.warning("Failed to download front image, continuing...")
                front_path = None
        
        # Upload images to ComfyUI
        logger.info("Step 2: Uploading images to ComfyUI...")
        
        uploaded_generated = comfyui_client.upload_image(generated_path)
        if not uploaded_generated:
            return False, "Failed to upload generated image to ComfyUI"
        
        uploaded_original = comfyui_client.upload_image(original_path)
        if not uploaded_original:
            return False, "Failed to upload original image to ComfyUI"
        
        uploaded_reference_name = None
        if reference_path and reference_path.exists():
            uploaded_ref = comfyui_client.upload_image(reference_path)
            if uploaded_ref:
                uploaded_reference_name = uploaded_ref.get('name', reference_path.name)
        
        uploaded_front_name = None
        if front_path and front_path.exists():
            uploaded_front = comfyui_client.upload_image(front_path)
            if uploaded_front:
                uploaded_front_name = uploaded_front.get('name', front_path.name)
        
        # Modify workflow
        logger.info("Step 3: Generating workflow...")
        
        modified_workflow = WorkflowModifier.modify_workflow(
            workflow_template,
            generated_image=uploaded_generated.get('name', generated_path.name),
            original_image=uploaded_original.get('name', original_path.name),
            reference_image=uploaded_reference_name,
            front_image=uploaded_front_name,
            prompt=prompt,
            output_prefix=f"{row_dir}_result"
        )
        
        # Execute workflow
        logger.info("Step 4: Executing face swap workflow...")
        
        prompt_id = comfyui_client.queue_prompt(modified_workflow)
        if not prompt_id:
            return False, "Failed to queue workflow"
        
        logger.info(f"Waiting for completion (timeout: {COMFYUI_TIMEOUT}s)...")
        if not comfyui_client.wait_for_completion(prompt_id):
            return False, "Workflow execution failed or timed out"
        
        # Download results
        logger.info("Step 5: Downloading results...")
        
        output_images = comfyui_client.get_output_images(prompt_id)
        if not output_images:
            return False, "No output images generated"
        
        result_path = output_dir / "result.png"
        if not comfyui_client.download_output_image(output_images[0], result_path):
            return False, "Failed to download result image"
        
        logger.info(f"✓ Row {row_num} completed successfully")
        logger.info(f"  Output saved to: {result_path}")
        return True, None
        
    except Exception as e:
        logger.error(f"Error processing row {row_num}: {e}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Automated face swap pipeline using ComfyUI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV file with image URLs and prompts'
    )
    parser.add_argument(
        '--workflow',
        required=True,
        help='Path to ComfyUI workflow JSON template'
    )
    parser.add_argument(
        '--comfyui-url',
        default=DEFAULT_COMFYUI_URL,
        help=f'ComfyUI server URL (default: {DEFAULT_COMFYUI_URL})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Number of rows to process before git commit (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--start-row',
        type=int,
        default=1,
        help='Start processing from this row number (1-indexed, default: 1)'
    )
    parser.add_argument(
        '--end-row',
        type=int,
        help='End processing at this row number (inclusive, default: all rows)'
    )
    parser.add_argument(
        '--no-git',
        action='store_true',
        help='Disable git commits and pushes'
    )
    parser.add_argument(
        '--gpu-ids',
        type=str,
        help='Comma-separated GPU IDs to use (e.g., "0,1,2,3" for 4 H100s)'
    )
    
    args = parser.parse_args()
    
    # Set GPU environment variables if specified
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        logger.info(f"Using GPUs: {args.gpu_ids}")
    
    # Validate inputs
    csv_path = Path(args.csv)
    workflow_path = Path(args.workflow)
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    if not workflow_path.exists():
        logger.error(f"Workflow file not found: {workflow_path}")
        sys.exit(1)
    
    # Load CSV
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Found {len(df)} rows in CSV")
    
    # Filter rows
    start_idx = args.start_row - 1  # Convert to 0-indexed
    end_idx = args.end_row if args.end_row else len(df)
    df_filtered = df.iloc[start_idx:end_idx]
    
    logger.info(f"Processing rows {args.start_row} to {min(end_idx, len(df))}")
    
    # Load workflow template
    logger.info(f"Loading workflow template: {workflow_path}")
    with open(workflow_path, 'r') as f:
        workflow_template = json.load(f)
    
    # Setup directories
    input_base_dir = Path(INPUT_DIR)
    output_base_dir = Path(OUTPUT_DIR)
    input_base_dir.mkdir(parents=True, exist_ok=True)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ComfyUI client
    comfyui_client = ComfyUIClient(args.comfyui_url)
    logger.info(f"Connected to ComfyUI at {args.comfyui_url}")
    
    # Process rows
    results = []
    batch_count = 0
    batch_start_row = args.start_row
    
    for idx, row in df_filtered.iterrows():
        row_num = idx + 1  # 1-indexed row number
        
        success, error = process_row(
            row,
            row_num,
            workflow_template,
            comfyui_client,
            input_base_dir,
            output_base_dir
        )
        
        results.append({
            'row': row_num,
            'success': success,
            'error': error,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        batch_count += 1
        
        # Batch git commit
        if not args.no_git and success and batch_count >= args.batch_size:
            successful = sum(1 for r in results[-batch_count:] if r['success'])
            failed = batch_count - successful
            batch_end_row = row_num
            
            commit_msg = f"Process rows {batch_start_row}-{batch_end_row}: {successful} successful, {failed} failed"
            logger.info(f"\nCommitting batch: {commit_msg}")
            
            GitManager.commit_and_push(commit_msg)
            
            batch_count = 0
            batch_start_row = row_num + 1
    
    # Final batch commit
    if not args.no_git and batch_count > 0:
        successful = sum(1 for r in results[-batch_count:] if r['success'])
        failed = batch_count - successful
        batch_end_row = args.start_row + len(df_filtered) - 1
        
        commit_msg = f"Process rows {batch_start_row}-{batch_end_row}: {successful} successful, {failed} failed"
        logger.info(f"\nCommitting final batch: {commit_msg}")
        
        GitManager.commit_and_push(commit_msg)
    
    # Save processing log
    log_path = Path("processing_log.json")
    with open(log_path, 'w') as f:
        json.dump({
            'summary': {
                'total_rows': len(results),
                'successful': sum(1 for r in results if r['success']),
                'failed': sum(1 for r in results if not r['success']),
                'start_time': results[0]['timestamp'] if results else None,
                'end_time': results[-1]['timestamp'] if results else None
            },
            'results': results
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Processing Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {len(results)} rows")
    logger.info(f"Successful: {sum(1 for r in results if r['success'])}")
    logger.info(f"Failed: {sum(1 for r in results if not r['success'])}")
    logger.info(f"Log saved to: {log_path}")
    
    # Exit with error code if any failures
    if any(not r['success'] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
