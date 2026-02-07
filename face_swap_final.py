#!/usr/bin/env python3
"""
Face Swap Automation with Auto Face Mask Generation
Uses MediaPipe for face detection and creates masked PNGs for ComfyUI
Tracks quality metrics, timing, and updates CSV with results
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Iterable

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image

DEFAULT_COMFYUI_URL = "http://localhost:8189"
DEFAULT_WORKFLOW_BASE = "Flux2 Klein 9b Face Swap(API)"
MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = 30
COMFYUI_TIMEOUT = 600
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

QUALITY_WORKFLOW = {
    "fast": f"{DEFAULT_WORKFLOW_BASE}.json",
    "balanced": f"{DEFAULT_WORKFLOW_BASE}.json",
    "high": f"{DEFAULT_WORKFLOW_BASE}.json",
    "ultra": f"{DEFAULT_WORKFLOW_BASE}.json",
}

QUALITY_PRESETS = {
    "fast": {"steps": 4, "denoise": 0.8, "cfg": 1, "lora_strength": 1.0, "megapixels": 2},
    "balanced": {"steps": 8, "denoise": 0.9, "cfg": 2, "lora_strength": 0.95, "megapixels": 4},
    "high": {"steps": 12, "denoise": 0.95, "cfg": 3, "lora_strength": 0.9, "megapixels": 6},
    "ultra": {"steps": 20, "denoise": 1.0, "cfg": 4, "lora_strength": 0.85, "megapixels": 8.3},
}

DEFAULT_ROW_PROMPT = (
    "Use image 1 (Original Image) as the strict base for full scene composition: keep its background, "
    "location, camera framing, pose, gesture, hand signs, body orientation, lighting, and perspective. "
    "Transfer only character identity details from image 2 (Generated Image): face identity and body shape. "
    "Do not move the subject to a new location, do not change gesture/pose from image 1, and do not alter "
    "camera angle from image 1. Keep output photorealistic and clean."
)


def apply_quality_preset(workflow: Dict, quality: str) -> None:
    params = QUALITY_PRESETS.get(quality)
    if not params:
        return
    if "156" in workflow:
        workflow["156"]["inputs"]["steps"] = params["steps"]
        workflow["156"]["inputs"]["denoise"] = params["denoise"]
    if "100" in workflow:
        workflow["100"]["inputs"]["guidance"] = params["cfg"]
    if "161" in workflow:
        workflow["161"]["inputs"]["strength_model"] = params["lora_strength"]
    if "135" in workflow:
        workflow["135"]["inputs"]["value"] = params["megapixels"]
    logger.info(f"Quality preset: {quality} (steps={params['steps']}, denoise={params['denoise']}, cfg={params['cfg']}, {params['megapixels']}MP)")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('face_swap.log')
    ]
)
logger = logging.getLogger(__name__)

def get_first_available_value(row_data: pd.Series, candidate_columns: Iterable[str]) -> str:
    """Return first non-empty value from matching column names."""
    normalized_map = {str(c).strip().lower(): c for c in row_data.index}
    for candidate in candidate_columns:
        key = normalized_map.get(candidate.strip().lower())
        if key is None:
            continue
        value = row_data.get(key, "")
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def extract_image_url(raw_value: Any) -> str:
    """Extract URL from plain text or =IMAGE(\"url\") formula cells."""
    if raw_value is None:
        return ""
    text = str(raw_value).strip()
    if not text:
        return ""
    if text.startswith("=IMAGE("):
        match = re.search(r'"([^"]+)"', text)
        if match:
            return match.group(1).strip()
    return text


def build_row_prompt(edit_prompt: str) -> str:
    base = DEFAULT_ROW_PROMPT
    if not edit_prompt:
        return base
    return f"{base}\nAdditional instruction: {edit_prompt.strip()}"


def get_image_metrics(image_path: Path) -> Dict[str, Any]:
    """Get image quality metrics"""
    try:
        img = Image.open(image_path)
        file_size = image_path.stat().st_size
        width, height = img.size
        
        arr = np.array(img.convert('RGB'))
        brightness = arr.mean()
        contrast = arr.std()
        
        return {
            'width': width,
            'height': height,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'megapixels': round((width * height) / 1_000_000, 2),
            'brightness': round(brightness, 2),
            'contrast': round(contrast, 2),
            'mode': img.mode
        }
    except Exception as e:
        logger.error(f"Failed to get metrics for {image_path}: {e}")
        return {}


def detect_face_opencv(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect face using OpenCV's Haar cascade (fallback method)"""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return (x, y, w, h)
    except Exception as e:
        logger.warning(f"OpenCV face detection failed: {e}")
    return None


def detect_face_mediapipe(img: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
    """Detect face using MediaPipe (returns x, y, w, h, confidence)"""
    try:
        import mediapipe as mp
        if not hasattr(mp, 'solutions'):
            return None
        
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb)
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                face_w = int(bbox.width * w)
                face_h = int(bbox.height * h)
                conf = detection.score[0]
                return (x1, y1, face_w, face_h, conf)
    except Exception as e:
        logger.warning(f"MediaPipe face detection failed: {e}")
    return None


def detect_face_and_create_mask(image_path: Path, output_path: Path, expansion: float = 0.3) -> Tuple[bool, Dict]:
    """
    Detect face in image and create a PNG with alpha mask for the face region.
    Tries MediaPipe first, falls back to OpenCV Haar cascade.
    Returns (success, face_info)
    """
    face_info = {'detected': False, 'confidence': 0, 'bbox': None, 'method': None}
    
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return False, face_info
        
        h, w = img.shape[:2]
        x1, y1, x2, y2 = 0, 0, 0, 0
        
        mp_result = detect_face_mediapipe(img)
        if mp_result:
            fx, fy, fw, fh, conf = mp_result
            face_info['detected'] = True
            face_info['confidence'] = round(conf, 3)
            face_info['bbox'] = {'x': fx, 'y': fy, 'width': fw, 'height': fh}
            face_info['method'] = 'mediapipe'
            
            exp_w = int(fw * expansion)
            exp_h = int(fh * expansion)
            x1 = max(0, fx - exp_w)
            y1 = max(0, fy - exp_h)
            x2 = min(w, fx + fw + exp_w)
            y2 = min(h, fy + fh + exp_h)
            logger.info(f"Face detected (MediaPipe): confidence={conf:.2f}")
        else:
            cv_result = detect_face_opencv(img)
            if cv_result:
                fx, fy, fw, fh = cv_result
                face_info['detected'] = True
                face_info['confidence'] = 0.8
                face_info['bbox'] = {'x': fx, 'y': fy, 'width': fw, 'height': fh}
                face_info['method'] = 'opencv'
                
                exp_w = int(fw * expansion)
                exp_h = int(fh * expansion)
                x1 = max(0, fx - exp_w)
                y1 = max(0, fy - exp_h)
                x2 = min(w, fx + fw + exp_w)
                y2 = min(h, fy + fh + exp_h)
                logger.info(f"Face detected (OpenCV Haar)")
            else:
                logger.warning(f"No face detected, using upper-center region")
                face_info['method'] = 'fallback'
                cx, cy = w // 2, h // 3
                face_w, face_h = w // 3, h // 3
                x1 = max(0, cx - face_w // 2)
                y1 = max(0, cy - face_h // 2)
                x2 = min(w, cx + face_w // 2)
                y2 = min(h, cy + face_h // 2)
        
        logger.info(f"Mask region: ({x1},{y1}) to ({x2},{y2})")
        
        mask = np.zeros((h, w), dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 10)
        
        pil_img = Image.open(image_path).convert('RGB')
        pil_mask = Image.fromarray(mask)
        pil_rgba = pil_img.copy()
        pil_rgba.putalpha(pil_mask)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_rgba.save(str(output_path), 'PNG')
        
        saved = Image.open(output_path)
        logger.info(f"Face mask created: {output_path} (mode={saved.mode})")
        return True, face_info
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False, face_info


def download_image(url: str, output_path: Path, retries: int = MAX_RETRIES) -> bool:
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


def modify_api_workflow(
    api_workflow: Dict,
    generated_img: str,
    original_img: str,
    output_prefix: str,
    row_prompt: str
) -> Dict:
    workflow = json.loads(json.dumps(api_workflow))
    
    if "151" in workflow:
        workflow["151"]["inputs"]["image"] = generated_img
        logger.info(f"Node 151: {generated_img}")
    
    if "121" in workflow:
        workflow["121"]["inputs"]["image"] = original_img  
        logger.info(f"Node 121: {original_img}")
    
    if "9" in workflow:
        workflow["9"]["inputs"]["filename_prefix"] = output_prefix
        logger.info(f"Node 9: {output_prefix}")

    if "107" in workflow and isinstance(workflow["107"], dict):
        workflow["107"].setdefault("inputs", {})
        workflow["107"]["inputs"]["text"] = row_prompt
        logger.info("Node 107: updated row prompt")
    
    return workflow


def queue_workflow(server_url: str, workflow: Dict, client_id: str) -> Optional[str]:
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
        if hasattr(e, 'response'):
            logger.error(f"Response: {e.response.text}")
        return None


def wait_for_completion(server_url: str, prompt_id: str, timeout: int = COMFYUI_TIMEOUT) -> bool:
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


def get_output_images(server_url: str, prompt_id: str, save_node_id: str = "9"):
    try:
        url = f"{server_url}/history/{prompt_id}"
        response = requests.get(url, timeout=10)
        history = response.json()
        
        images = []
        if prompt_id in history:
            outputs = history[prompt_id].get('outputs', {})
            
            if save_node_id in outputs and 'images' in outputs[save_node_id]:
                images = outputs[save_node_id]['images']
                logger.info(f"Found {len(images)} images from node {save_node_id}")
            else:
                for node_id, node_output in outputs.items():
                    if 'images' in node_output:
                        logger.info(f"Node {node_id} has {len(node_output['images'])} images")
                        images.extend(node_output['images'])
        
        return images
    except Exception as e:
        logger.error(f"Failed to get output images: {e}")
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
        
        logger.info(f"Saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def process_row(row_data: pd.Series, row_num: int, workflow_template: Dict, server_url: str, timeout: int = COMFYUI_TIMEOUT) -> Dict[str, Any]:
    """Process single CSV row and return detailed results"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Row {row_num}")
    logger.info(f"{'='*60}")
    
    row_dir = f"row_{row_num:04d}"
    input_dir = Path(INPUT_DIR) / row_dir
    output_dir = Path(OUTPUT_DIR) / row_dir
    
    result = {
        'row': row_num,
        'success': False,
        'timestamp': datetime.now().isoformat(),
        'generated_url': '',
        'original_url': '',
        'edit_prompt': '',
        'output_path': '',
        'timing': {
            'download_sec': 0,
            'face_detection_sec': 0,
            'upload_sec': 0,
            'processing_sec': 0,
            'total_sec': 0
        },
        'face_detection': {},
        'input_metrics': {},
        'output_metrics': {},
        'error': None
    }
    
    total_start = time.time()
    
    try:
        generated_url = extract_image_url(
            get_first_available_value(row_data, ["Generated Image", "Reference Angle", "Front Angle"])
        )
        original_url = extract_image_url(get_first_available_value(row_data, ["Original Image"]))
        edit_prompt = get_first_available_value(row_data, ["Edit Prompt", "edit prompt", "Prompt"])
        result['generated_url'] = generated_url
        result['original_url'] = original_url
        result['edit_prompt'] = edit_prompt
        
        if not generated_url or not original_url:
            result['error'] = "Missing image URLs"
            logger.error(result['error'])
            return result
        
        # Download
        download_start = time.time()
        logger.info("Downloading...")
        gen_path = input_dir / "generated_raw.png"
        orig_path = input_dir / "original.png"
        
        if not download_image(generated_url, gen_path):
            result['error'] = "Failed to download generated image"
            return result
        if not download_image(original_url, orig_path):
            result['error'] = "Failed to download original image"
            return result
        result['timing']['download_sec'] = round(time.time() - download_start, 2)
        
        # Get input metrics
        result['input_metrics']['generated'] = get_image_metrics(gen_path)
        result['input_metrics']['original'] = get_image_metrics(orig_path)
        
        # Face detection
        face_start = time.time()
        logger.info("Detecting face and creating mask...")
        gen_masked_path = input_dir / "generated.png"
        mask_success, face_info = detect_face_and_create_mask(gen_path, gen_masked_path)
        result['face_detection'] = face_info
        
        if not mask_success:
            logger.warning("Face mask creation failed, using original image")
            gen_masked_path = gen_path
        result['timing']['face_detection_sec'] = round(time.time() - face_start, 2)
        
        # Upload
        upload_start = time.time()
        logger.info("Uploading...")
        gen_name = upload_to_comfyui(server_url, gen_masked_path)
        orig_name = upload_to_comfyui(server_url, orig_path)
        
        if not gen_name or not orig_name:
            result['error'] = "Failed to upload images"
            return result
        result['timing']['upload_sec'] = round(time.time() - upload_start, 2)
        
        # Process
        process_start = time.time()
        logger.info("Preparing workflow...")
        modified = modify_api_workflow(
            workflow_template,
            generated_img=gen_name,
            original_img=orig_name,
            output_prefix=f"{row_dir}_result",
            row_prompt=build_row_prompt(edit_prompt),
        )
        
        logger.info("Executing face swap...")
        client_id = str(int(time.time() * 1000))
        prompt_id = queue_workflow(server_url, modified, client_id)
        
        if not prompt_id:
            result['error'] = "Failed to queue workflow"
            return result
        
        if not wait_for_completion(server_url, prompt_id, timeout=timeout):
            result['error'] = "Workflow execution failed or timed out"
            return result
        
        logger.info("Downloading results...")
        images = get_output_images(server_url, prompt_id)
        if not images:
            result['error'] = "No output images returned"
            return result
        
        result_path = output_dir / "result.png"
        if not download_output(server_url, images[0], result_path):
            result['error'] = "Failed to download result"
            return result
        
        result['timing']['processing_sec'] = round(time.time() - process_start, 2)
        
        # Get output metrics
        result['output_path'] = str(result_path)
        result['output_metrics'] = get_image_metrics(result_path)
        
        result['success'] = True
        logger.info(f"SUCCESS: {result_path}")
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    result['timing']['total_sec'] = round(time.time() - total_start, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description='Face swap automation using ComfyUI API')
    parser.add_argument('--csv', required=True, help='CSV file path')
    parser.add_argument('--workflow', help='API format workflow JSON (overrides --quality if set)')
    parser.add_argument('--quality', choices=list(QUALITY_WORKFLOW), help='Quality preset: fast, balanced, high, ultra (uses default workflow files)')
    parser.add_argument('--comfyui-url', default=DEFAULT_COMFYUI_URL, help='ComfyUI server URL')
    parser.add_argument('--start-row', type=int, default=1, help='Start row (1-indexed)')
    parser.add_argument('--end-row', type=int, help='End row (inclusive)')
    parser.add_argument('--gpu-ids', type=str, help='GPU IDs (e.g., 0,1,2,3)')
    parser.add_argument('--update-csv', action='store_true', help='Update CSV with results columns')
    parser.add_argument('--output-csv', type=str, help='Output CSV path (default: adds _results suffix)')
    parser.add_argument(
        '--minimal-csv',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Write output CSV with only Original Image, Generated Image, new image (default: true)'
    )
    parser.add_argument('--timeout', type=int, default=COMFYUI_TIMEOUT, help=f'ComfyUI wait timeout in seconds (default: {COMFYUI_TIMEOUT})')
    
    args = parser.parse_args()
    if not args.workflow and not args.quality:
        parser.error("Either --workflow or --quality is required")
    workflow_path = args.workflow or QUALITY_WORKFLOW[args.quality]
    if not os.path.isabs(workflow_path) and not Path(workflow_path).exists():
        workflow_path = str(Path(__file__).resolve().parent / Path(workflow_path).name)
    args.workflow = workflow_path

    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        logger.info(f"Using GPUs: {args.gpu_ids}")
    
    df = pd.read_csv(args.csv)
    logger.info(f"CSV: {len(df)} rows")
    
    with open(args.workflow) as f:
        workflow_template = json.load(f)
    logger.info(f"Workflow: {args.workflow}")
    logger.info(f"Nodes in workflow: {len(workflow_template)}")
    if args.quality:
        apply_quality_preset(workflow_template, args.quality)
    
    end = args.end_row if args.end_row else len(df)
    results = []
    run_start = datetime.now()
    
    # Add new columns if updating CSV
    if args.update_csv:
        if args.minimal_csv:
            if 'new image' not in df.columns:
                df['new image'] = ''
        else:
            if 'Swap_Status' not in df.columns:
                df['Swap_Status'] = ''
            if 'Swap_Output_Path' not in df.columns:
                df['Swap_Output_Path'] = ''
            if 'Swap_Time_Sec' not in df.columns:
                df['Swap_Time_Sec'] = 0.0
            if 'Swap_Output_Size_MB' not in df.columns:
                df['Swap_Output_Size_MB'] = 0.0
            if 'Swap_Output_Width' not in df.columns:
                df['Swap_Output_Width'] = 0
            if 'Swap_Output_Height' not in df.columns:
                df['Swap_Output_Height'] = 0
            if 'Swap_Face_Detected' not in df.columns:
                df['Swap_Face_Detected'] = False
            if 'Swap_Face_Confidence' not in df.columns:
                df['Swap_Face_Confidence'] = 0.0
    
    for idx in range(args.start_row - 1, min(end, len(df))):
        row_num = idx + 1
        result = process_row(df.iloc[idx], row_num, workflow_template, args.comfyui_url, args.timeout)
        results.append(result)
        
        # Update DataFrame
        if args.update_csv:
            if args.minimal_csv:
                df.at[idx, 'new image'] = result.get('output_path', '') if result['success'] else ''
            else:
                df.at[idx, 'Swap_Status'] = 'success' if result['success'] else f"failed: {result.get('error', 'unknown')}"
                df.at[idx, 'Swap_Output_Path'] = result.get('output_path', '')
                df.at[idx, 'Swap_Time_Sec'] = result['timing']['total_sec']
                df.at[idx, 'Swap_Output_Size_MB'] = result.get('output_metrics', {}).get('file_size_mb', 0)
                df.at[idx, 'Swap_Output_Width'] = result.get('output_metrics', {}).get('width', 0)
                df.at[idx, 'Swap_Output_Height'] = result.get('output_metrics', {}).get('height', 0)
                df.at[idx, 'Swap_Face_Detected'] = result.get('face_detection', {}).get('detected', False)
                df.at[idx, 'Swap_Face_Confidence'] = result.get('face_detection', {}).get('confidence', 0)
    
    run_end = datetime.now()
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_time = sum(r['timing']['total_sec'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {successful}/{len(results)} successful")
    logger.info(f"Total time: {total_time:.1f}s, Avg per row: {avg_time:.1f}s")
    logger.info(f"{'='*60}")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    results = [convert_to_native(r) for r in results]
    
    # Save comprehensive results.json
    results_data = {
        'run_info': {
            'start_time': run_start.isoformat(),
            'end_time': run_end.isoformat(),
            'duration_sec': (run_end - run_start).total_seconds(),
            'csv_file': args.csv,
            'workflow_file': args.workflow,
            'quality': getattr(args, 'quality', None),
            'comfyui_url': args.comfyui_url,
            'start_row': args.start_row,
            'end_row': end,
            'gpu_ids': args.gpu_ids,
            'timeout_sec': args.timeout
        },
        'summary': {
            'total_rows': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': round(successful / len(results) * 100, 1) if results else 0,
            'total_time_sec': round(total_time, 2),
            'avg_time_per_row_sec': round(avg_time, 2),
            'timing_breakdown': {
                'avg_download_sec': round(sum(r['timing']['download_sec'] for r in results) / len(results), 2) if results else 0,
                'avg_face_detection_sec': round(sum(r['timing']['face_detection_sec'] for r in results) / len(results), 2) if results else 0,
                'avg_upload_sec': round(sum(r['timing']['upload_sec'] for r in results) / len(results), 2) if results else 0,
                'avg_processing_sec': round(sum(r['timing']['processing_sec'] for r in results) / len(results), 2) if results else 0
            }
        },
        'results': results
    }
    
    with open("results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Results saved to: results.json")
    
    # Save updated CSV
    if args.update_csv:
        output_csv = args.output_csv or args.csv.replace('.csv', '_results.csv')
        if args.minimal_csv:
            for required_col in ['Original Image', 'Generated Image']:
                if required_col not in df.columns:
                    df[required_col] = ''
            out_df = df[['Original Image', 'Generated Image', 'new image']].copy()
            out_df.to_csv(output_csv, index=False)
        else:
            df.to_csv(output_csv, index=False)
        logger.info(f"Updated CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
