# Character Face Swap Automation

Automated batch processing pipeline for face swapping using ComfyUI Flux2 Klein 9b workflow with MediaPipe face detection.

## Overview

Production-ready face swap automation that:
- Processes CSV data with image URLs and prompts
- Automatically detects faces using MediaPipe (with OpenCV fallback)
- Executes face swaps via ComfyUI API
- Tracks quality metrics, timing, and processing stats
- Updates CSV with results and Azure CDN URLs
- Supports resolution testing (1K/2K/4K)

## Prerequisites

### GPU Server Requirements
- Python 3.10+
- CUDA-enabled GPUs (tested on 4x H100)
- ComfyUI installed and running
- Git configured
- Internet access for downloading images

### Required Models
- **UNET**: `flux-2-klein-9b.safetensors` (~18GB)
- **VAE**: `flux2-vae.safetensors` (~335MB)
- **CLIP**: `split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors` (~3GB)
- **LoRA**: `bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors`

### Custom Nodes
- LanPaint (for inpainting)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ch33nchan/character-swap.git
cd character-swap
```

### 2. Setup ComfyUI

Run the automated setup script:

```bash
chmod +x setup_comfyui.sh
./setup_comfyui.sh
```

Or install manually:

```bash
# Install ComfyUI
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install dependencies
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt

# Install LanPaint custom node
cd custom_nodes
git clone https://github.com/scraed/LanPaint.git
cd LanPaint
pip3 install -r requirements.txt
```

### 3. Download Models

```bash
cd ~/ComfyUI/models

# Flux2 Klein 9b UNET
mkdir -p unet && cd unet
wget --header="Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors

# VAE + Text Encoder
cd .. && mkdir -p vae clip
cd vae
wget https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors

cd ../clip
wget https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors

# LoRA
cd .. && mkdir -p loras && cd loras
wget https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors
```

### 4. Setup Project Environment

```bash
cd ~/character-swap
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### 5. Start ComfyUI

```bash
# Create separate venv for ComfyUI (avoids NumPy conflicts)
cd ~/ComfyUI
python3 -m venv comfyui_venv
source comfyui_venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Start server with all GPUs
python main.py --listen 0.0.0.0 --port 8189 --highvram
```

Verify ComfyUI is running:

```bash
curl http://localhost:8189/system_stats
```

## Usage

### Basic Face Swap Processing

Process all rows in CSV:

```bash
cd ~/character-swap
source venv/bin/activate

python3 face_swap_final.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap(API).json" \
    --comfyui-url http://localhost:8189
```

Test run (first 5 rows):

```bash
python3 face_swap_final.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap(API).json" \
    --start-row 1 \
    --end-row 5
```

Update CSV with results:

```bash
python3 face_swap_final.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap(API).json" \
    --update-csv \
    --output-csv "results_with_metrics.csv"
```

### Resolution Testing

Test performance at 1K, 2K, and 4K resolutions:

```bash
python3 test_resolutions.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap(API).json" \
    --row 1 \
    --comfyui-url http://localhost:8189
```

Output:
- `resolution_test_row1/1K.png` - 1 megapixel result
- `resolution_test_row1/2K.png` - 4 megapixel result
- `resolution_test_row1/4K.png` - 8.3 megapixel result
- `resolution_test_row1/results.json` - Timing and metrics

### Azure Upload & Google Sheets Formatting

Upload all result images to Azure Blob Storage and create Google Sheets IMAGE() formulas:

```bash
python3 upload_and_format.py
```

This will:
- Upload all `data/output/row_XXXX/result.png` to Azure
- Add `Swap_Azure_URL` column with CDN URLs
- Add `Swap_Image_Formula` column with `=IMAGE("url")`
- Convert all existing URL columns to formula columns
- Save to `Master - Tech Solutioning - Char Const - Rerun with head_eye gaze_results_with_azure_urls.csv`

## Command-Line Arguments

### face_swap_final.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | - | Path to CSV file with image URLs and prompts |
| `--workflow` | Yes | - | Path to ComfyUI API workflow JSON |
| `--comfyui-url` | No | `http://localhost:8189` | ComfyUI server URL |
| `--start-row` | No | `1` | Start processing from this row (1-indexed) |
| `--end-row` | No | All rows | End processing at this row (inclusive) |
| `--update-csv` | No | False | Update input CSV with results |
| `--output-csv` | No | Auto-generated | Custom output CSV path |

### test_resolutions.py

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | - | Path to CSV file |
| `--workflow` | Yes | - | Path to API workflow JSON |
| `--row` | No | `1` | Row number to test (1-indexed) |
| `--comfyui-url` | No | `http://localhost:8189` | ComfyUI server URL |

## CSV Format

Required columns:
- `Original Image` - URL to base image
- `Generated Image` - URL to AI-generated character image
- `Edit Prompt` - Detailed face swap instructions
- `Reference Angle` (optional) - URL to reference angle image
- `Front Angle` (optional) - URL to front angle image

Output columns (when using `--update-csv`):
- `Swap_Status` - "success" or "failed"
- `Swap_Output_Path` - Local path to result image
- `Swap_Time_Sec` - Processing time in seconds
- `Swap_Output_Size_MB` - File size in MB
- `Swap_Output_Width` - Image width in pixels
- `Swap_Output_Height` - Image height in pixels
- `Swap_Face_Detected` - Whether face was detected (True/False)
- `Swap_Face_Confidence` - Face detection confidence (0-1)

## Output Structure

```
character-swap/
├── face_swap_final.py              # Main production script
├── test_resolutions.py             # Resolution testing
├── upload_and_format.py            # Azure upload + IMAGE() formulas
├── requirements.txt                # Python dependencies
├── setup_comfyui.sh               # Setup script
├── Flux2 Klein 9b Face Swap(API).json  # Working workflow
├── Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv
├── Master - Tech Solutioning - Char Const - Rerun with head_eye gaze_results.csv
├── results.json                    # Processing results metadata
├── face_swap.log                   # Execution log
├── .gitignore                      # Git config
├── venv/                          # Python environment
└── data/
    └── output/                    # All result images
        ├── row_0001/
        │   └── result.png
        ├── row_0002/
        └── ...
```

## Processing Results

After completion, `results.json` contains:

```json
{
  "run_info": {
    "start_time": "2026-02-05T10:57:30",
    "end_time": "2026-02-05T11:00:04",
    "duration_sec": 153.78,
    "csv_file": "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv",
    "workflow_file": "Flux2 Klein 9b Face Swap(API).json",
    "comfyui_url": "http://localhost:8189"
  },
  "summary": {
    "total_rows": 49,
    "successful": 48,
    "failed": 1,
    "success_rate": 97.96,
    "total_time_sec": 153.78,
    "avg_time_per_row_sec": 3.14,
    "timing_breakdown": {
      "avg_download_sec": 0.51,
      "avg_face_detection_sec": 0.16,
      "avg_upload_sec": 0.11,
      "avg_processing_sec": 29.56
    }
  },
  "results": [...]
}
```

## Features

### Automatic Face Detection
- Uses MediaPipe for accurate face detection
- Falls back to OpenCV Haar cascade if MediaPipe unavailable
- Creates RGBA PNG masks with alpha channel for face region
- Expands face region by 30% for better coverage

### Error Handling
- Download failures: Retries up to 3 times, then skips row
- Face detection failures: Uses fallback upper-center region
- ComfyUI execution failures: Logs error, continues to next row
- All errors logged with full details in `results.json`

### Quality Metrics
- Image dimensions (width x height)
- File size (MB)
- Megapixels
- Brightness and contrast measurements
- Face detection confidence scores

### Azure Integration
- Uploads to production storage account (`dashprodstore`)
- Uses CDN for fast access (`https://content.dashtoon.ai`)
- Organized blob structure: `face-swap-results/row_XXXX/result.png`
- Generates Google Sheets `=IMAGE()` formulas

## Troubleshooting

### ComfyUI Connection Error

```bash
# Check if ComfyUI is running
curl http://localhost:8189/system_stats

# Check process
ps aux | grep "python.*main.py"

# Restart ComfyUI
cd ~/ComfyUI
source comfyui_venv/bin/activate
python main.py --listen 0.0.0.0 --port 8189 --highvram
```

### NumPy Version Conflicts

ComfyUI requires specific NumPy versions. Solution: Use separate venvs.

```bash
# ComfyUI venv (with compatible NumPy)
cd ~/ComfyUI
python3 -m venv comfyui_venv
source comfyui_venv/bin/activate
pip install -r requirements.txt

# Character-swap venv (with latest NumPy for MediaPipe)
cd ~/character-swap
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Face Detection Not Working

If MediaPipe fails, the script automatically falls back to OpenCV:

```bash
# Verify MediaPipe installation
python3 -c "import mediapipe as mp; print(mp.solutions.face_detection)"

# If error, reinstall
pip install --force-reinstall mediapipe
```

### Black Output Images

This indicates the face mask wasn't created correctly. Check:

```bash
# Verify generated.png has alpha channel
python3 -c "from PIL import Image; img=Image.open('data/input/row_0001/generated.png'); print(img.mode)"
# Should output: RGBA
```

### Out of Memory

```bash
# Use normal VRAM mode instead of high
python main.py --listen 0.0.0.0 --port 8189 --normalvram

# Or reduce batch processing
python3 face_swap_final.py --end-row 10  # Process fewer rows at once
```

## Performance

### Expected Processing Time
- Image downloads: ~0.5 seconds per row
- Face detection: ~0.2 seconds per row
- ComfyUI processing: ~30 seconds per row (H100)
- Total per row: ~30-35 seconds

### For 49 Rows
- Estimated time: ~25-30 minutes
- Actual time (tested): ~26 minutes
- Success rate: 98%+

### Resolution Impact
- 1K (1MP): ~25 seconds per image
- 2K (4MP): ~30 seconds per image
- 4K (8.3MP): ~40 seconds per image

## Tips

### Long-Running Jobs

```bash
# Use tmux for persistent session
tmux new -s faceswap
python3 face_swap_final.py --csv "data.csv" --workflow "workflow.json"
# Detach: Ctrl+b, then d
# Reattach: tmux attach -t faceswap
```

### Monitor Progress

```bash
# Watch logs in real-time
tail -f face_swap.log

# Count completed rows
ls data/output/ | wc -l

# View results summary
cat results.json | jq '.summary'
```

### Resume from Specific Row

```bash
# If processing stopped at row 25
python3 face_swap_final.py \
    --csv "data.csv" \
    --workflow "workflow.json" \
    --start-row 25
```

## Repository

GitHub: https://github.com/ch33nchan/character-swap

## License

This project is for internal use only.

## Support

For issues or questions, check:
1. `face_swap.log` - Detailed execution log
2. `results.json` - Processing results and metrics
3. ComfyUI logs - Workflow execution details
4. GPU status: `nvidia-smi`
