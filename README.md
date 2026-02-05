# Character Face Swap Automation

Automated batch processing pipeline for face swapping using ComfyUI Flux2 Klein 9b workflow.

## Overview

This single-script solution processes CSV data containing image URLs and prompts, downloads images, executes face swaps via ComfyUI API, and automatically commits results to git.

## Prerequisites

**GPU Server Requirements:**
- Python 3.8+
- CUDA-enabled GPUs (tested on 4x H100)
- ComfyUI installed and running (see SETUP_GUIDE.md)
- Git configured
- Internet access for downloading images

**ComfyUI Setup:**
- Flux2 Klein 9b model (`flux-2-klein-9b.safetensors`)
- VAE model (`flux2-vae.safetensors`)
- CLIP model (`qwen_3_8b_fp8mixed.safetensors`)
- LoRA model (`bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors`)
- LanPaint custom node installed

## Quick Start

**First time setup? See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete installation instructions.**

### 1. Run Setup Script

```bash
chmod +x setup_comfyui.sh
./setup_comfyui.sh
```

This will verify/install ComfyUI and check all requirements.

### 2. Start ComfyUI with GPU Selection

```bash
cd ~/ComfyUI

# Use all 4 H100 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --listen 0.0.0.0 --port 8188 --highvram

# Or use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --listen 0.0.0.0 --port 8188
```

### 3. Setup Project Environment

```bash
cd character-swap
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Usage

### Basic Usage

Process all rows with 4 H100 GPUs:

```bash
python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --comfyui-url http://localhost:8188 \
    --gpu-ids 0,1,2,3
```

Test run (first 5 rows only):

```bash
python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --comfyui-url http://localhost:8188 \
    --gpu-ids 0,1,2,3 \
    --end-row 5 \
    --no-git
```

### Process Specific Rows

Process rows 1-50:

```bash
python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --start-row 1 \
    --end-row 50
```

### Custom Batch Size

Commit to git every 5 rows instead of default 10:

```bash
python3 face_swap.py \
    --csv "path/to/csv" \
    --workflow "path/to/workflow.json" \
    --batch-size 5
```

### Skip Git Operations

Process without git commits (useful for testing):

```bash
python3 face_swap.py \
    --csv "path/to/csv" \
    --workflow "path/to/workflow.json" \
    --no-git
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | Yes | - | Path to CSV file with image URLs and prompts |
| `--workflow` | Yes | - | Path to ComfyUI workflow JSON template |
| `--comfyui-url` | No | `http://localhost:8188` | ComfyUI server URL |
| `--batch-size` | No | `10` | Number of rows before git commit |
| `--start-row` | No | `1` | Start processing from this row (1-indexed) |
| `--end-row` | No | All rows | End processing at this row (inclusive) |
| `--no-git` | No | False | Disable git commits and pushes |
| `--gpu-ids` | No | All GPUs | Comma-separated GPU IDs (e.g., "0,1,2,3") |

## CSV Format

Required columns in CSV:
- `Original Image` - URL to base image
- `Generated Image` - URL to AI-generated character image
- `Edit Prompt` - Detailed face swap instructions
- `Reference Angle` (optional) - URL to reference angle image
- `Front Angle` (optional) - URL to front angle image

## Output Structure

```
character-swap/
├── data/
│   ├── input/              # Downloaded source images
│   │   ├── row_0001/
│   │   │   ├── generated.png
│   │   │   ├── original.png
│   │   │   ├── reference.png  (optional)
│   │   │   └── front.png      (optional)
│   │   ├── row_0002/
│   │   └── ...
│   └── output/             # Face-swapped results
│       ├── row_0001/
│       │   └── result.png
│       ├── row_0002/
│       └── ...
├── processing_log.json     # Detailed processing log
└── face_swap.log           # Execution log
```

## Processing Log

After completion, `processing_log.json` contains:

```json
{
  "summary": {
    "total_rows": 100,
    "successful": 95,
    "failed": 5,
    "start_time": "2026-02-05 10:00:00",
    "end_time": "2026-02-05 12:30:00"
  },
  "results": [
    {
      "row": 1,
      "success": true,
      "error": null,
      "timestamp": "2026-02-05 10:05:23"
    }
  ]
}
```

## Error Handling

The script automatically handles common errors:

- **Download failures**: Skips row, logs error, continues to next
- **ComfyUI execution failures**: Skips row, logs error, continues to next
- **Network timeouts**: Retries up to 3 times with exponential backoff
- **Git push failures**: Logs error, continues processing

Failed rows are logged in `processing_log.json` with error details.

## Monitoring Progress

**Real-time logs:**
```bash
tail -f face_swap.log
```

**Check progress:**
```bash
# Count completed rows
ls data/output/ | wc -l

# View processing log
cat processing_log.json
```

## Workflow Modifications

The script dynamically modifies these ComfyUI workflow nodes:

| Node ID | Type | Purpose | Parameter |
|---------|------|---------|-----------|
| 151 | LoadImage | Generated image input | `widgets_values[0]` |
| 121 | LoadImage | Original image input | `widgets_values[0]` |
| 128 | LoadImage | Reference angle (optional) | `widgets_values[0]` |
| 137 | LoadImage | Front angle (optional) | `widgets_values[0]` |
| 107 | CLIPTextEncode | Edit prompt | `widgets_values[0]` |
| 9 | SaveImage | Output filename | `widgets_values[0]` |

## Troubleshooting

### ComfyUI Connection Error

```
Error: Failed to connect to ComfyUI at http://localhost:8188
```

**Solution:** Verify ComfyUI is running:
```bash
curl http://localhost:8188/system_stats
```

### Image Download Failures

```
Error: Failed to download https://...
```

**Solution:** Check internet connectivity and URL accessibility

### Workflow Execution Timeout

```
Error: Workflow execution failed or timed out
```

**Solution:** 
- Check ComfyUI logs for errors
- Verify all required models are installed
- Increase timeout in script (edit `COMFYUI_TIMEOUT` constant)

### Git Push Failures

```
Error: Git push failed
```

**Solution:**
- Verify git credentials are configured
- Check repository permissions
- Use `--no-git` flag to skip git operations

## Performance

**Expected processing time per row:**
- Image downloads: ~10-30 seconds
- ComfyUI face swap: ~30-120 seconds (GPU-dependent)
- Total per row: ~1-3 minutes

**For 1000 rows:**
- Estimated time: 16-50 hours
- Recommended: Run in `tmux` or `screen` session

## Tips

**Long-running jobs:**
```bash
# Use tmux for persistent session
tmux new -s faceswap
python3 face_swap.py --csv "data.csv" --workflow "workflow.json"
# Detach: Ctrl+b, then d
# Reattach: tmux attach -t faceswap
```

**Resume from specific row:**
```bash
# If processing stopped at row 150
python3 face_swap.py \
    --csv "data.csv" \
    --workflow "workflow.json" \
    --start-row 150
```

**Test on small subset:**
```bash
# Test on first 5 rows
python3 face_swap.py \
    --csv "data.csv" \
    --workflow "workflow.json" \
    --end-row 5 \
    --no-git
```

## License

This project is for internal use only.

## Support

For issues or questions, check:
1. `face_swap.log` - Detailed execution log
2. `processing_log.json` - Processing results
3. ComfyUI logs - Workflow execution details
