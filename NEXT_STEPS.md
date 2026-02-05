# Next Steps - GPU Server Deployment

Successfully pushed to: https://github.com/ch33nchan/character-swap

## Step-by-Step Deployment Guide

### 1. SSH to Your GPU Server

```bash
ssh your-username@your-gpu-server
```

### 2. Clone the Repository

```bash
cd ~
git clone https://github.com/ch33nchan/character-swap.git
cd character-swap
```

### 3. Run Setup Verification

This will check your system and ComfyUI installation:

```bash
chmod +x setup_comfyui.sh
./setup_comfyui.sh
```

**What the script checks:**
- ✓ Python 3 installation
- ✓ CUDA and GPU availability (should detect your 4 H100s)
- ✓ ComfyUI installation (will offer to install if missing)
- ✓ Required models (Flux2 Klein 9b, VAE, CLIP, LoRA)
- ✓ Custom nodes (LanPaint)

**If ComfyUI is not installed**, the script will guide you through installation.

### 4. Download Required Models (If Missing)

If the setup script reports missing models, download them:

```bash
cd ~/ComfyUI/models

# Flux2 Klein 9b UNET (~18GB)
mkdir -p unet && cd unet
wget https://huggingface.co/Comfy-Org/flux2-klein-9b/resolve/main/flux-2-klein-9b.safetensors

# VAE (~335MB)
cd .. && mkdir -p vae && cd vae
wget https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors

# CLIP (~3GB)
cd .. && mkdir -p clip && cd clip
wget https://huggingface.co/Comfy-Org/qwen-2.5-1.5b-t2v-text-encoder/resolve/main/qwen_3_8b_fp8mixed.safetensors

# LoRA (custom - get from your source)
cd .. && mkdir -p loras
# Copy bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors here
```

### 5. Install LanPaint Custom Node (If Missing)

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/scraed/LanPaint.git
cd LanPaint
pip3 install -r requirements.txt
```

### 6. Setup Project Environment

```bash
cd ~/character-swap
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### 7. Start ComfyUI with All 4 H100s

Open a new terminal or tmux session:

```bash
# Start tmux for persistent session
tmux new -s comfyui

# Navigate to ComfyUI
cd ~/ComfyUI

# Start with all 4 H100 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --listen 0.0.0.0 --port 8188 --highvram

# Detach from tmux: Ctrl+b, then d
```

**Verify ComfyUI is running:**

```bash
curl http://localhost:8188/system_stats
# Should return JSON with system info
```

### 8. Test Face Swap on 5 Rows

Before running the full batch, test on a small subset:

```bash
cd ~/character-swap
source venv/bin/activate

python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --comfyui-url http://localhost:8188 \
    --gpu-ids 0,1,2,3 \
    --end-row 5 \
    --no-git
```

**Check results:**

```bash
ls -la data/output/row_0001/
cat processing_log.json
```

### 9. Run Full Production Batch

If test successful, run the full pipeline:

```bash
# Start another tmux session for the face swap pipeline
tmux new -s faceswap

cd ~/character-swap
source venv/bin/activate

# Process all 1128 rows with git auto-commit
python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --comfyui-url http://localhost:8188 \
    --gpu-ids 0,1,2,3 \
    --batch-size 10

# Detach: Ctrl+b, then d
```

### 10. Monitor Progress

```bash
# Reattach to face swap session
tmux attach -t faceswap

# Or watch logs
tail -f ~/character-swap/face_swap.log

# Check completed rows
ls ~/character-swap/data/output/ | wc -l

# View processing summary
cat ~/character-swap/processing_log.json | jq '.summary'
```

## Performance Expectations

With 4x H100 GPUs:
- **Per image**: 1-2 minutes (includes download, face swap, save)
- **Total (1128 images)**: ~20-35 hours
- **Git commits**: Every 10 rows automatically

## GPU Configuration Options

### All 4 H100s (Recommended)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --listen 0.0.0.0 --port 8188 --highvram
python3 face_swap.py --gpu-ids 0,1,2,3 ...
```

### 2 GPUs (If memory issues)
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --listen 0.0.0.0 --port 8188
python3 face_swap.py --gpu-ids 0,1 ...
```

### Single GPU (Conservative)
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --listen 0.0.0.0 --port 8188
python3 face_swap.py --gpu-ids 0 ...
```

## Troubleshooting Common Issues

### Issue 1: ComfyUI Connection Error

```bash
# Check if ComfyUI is running
ps aux | grep "python3 main.py"

# Check port
netstat -tulpn | grep 8188

# Try restarting ComfyUI
tmux attach -t comfyui
# Ctrl+C to stop, then restart
python3 main.py --listen 0.0.0.0 --port 8188 --highvram
```

### Issue 2: CUDA Out of Memory

```bash
# Use normal VRAM mode instead of high
python3 main.py --listen 0.0.0.0 --port 8188 --normalvram

# Or reduce GPU count
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --listen 0.0.0.0 --port 8188
```

### Issue 3: Models Not Found

```bash
# Check model directory
ls -lh ~/ComfyUI/models/unet/
ls -lh ~/ComfyUI/models/vae/
ls -lh ~/ComfyUI/models/clip/
ls -lh ~/ComfyUI/models/loras/

# Verify paths in ComfyUI
cd ~/ComfyUI
python3 -c "import folder_paths; print(folder_paths.folder_names_and_paths)"
```

### Issue 4: Image Download Failures

```bash
# Check internet connectivity
curl -I https://content.dashtoon.ai/

# Check DNS
nslookup content.dashtoon.ai

# Retry failed rows
python3 face_swap.py --start-row <failed_row> --end-row <failed_row> ...
```

### Issue 5: Git Push Failures

```bash
# Configure git credentials
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Setup SSH key or credential helper
git config --global credential.helper store

# Or disable git operations
python3 face_swap.py --no-git ...
```

## Resuming After Interruption

If the pipeline stops or crashes:

```bash
# Find last completed row
ls ~/character-swap/data/output/ | tail -n 1
# Example output: row_0150

# Resume from next row
python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --start-row 151 \
    --gpu-ids 0,1,2,3
```

## Monitoring Commands

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Disk space
df -h

# Processing progress
tail -f face_swap.log | grep "Processing Row"

# Success rate
cat processing_log.json | jq '.summary'

# Failed rows
cat processing_log.json | jq '.results[] | select(.success == false)'
```

## Expected Output Structure

```
character-swap/
├── data/
│   ├── input/
│   │   ├── row_0001/
│   │   │   ├── generated.png
│   │   │   ├── original.png
│   │   │   ├── reference.png
│   │   │   └── front.png
│   │   └── row_0002/...
│   └── output/
│       ├── row_0001/
│       │   └── result.png
│       └── row_0002/...
├── face_swap.log
└── processing_log.json
```

## Final Checklist

- [ ] Cloned repository from GitHub
- [ ] Ran `setup_comfyui.sh` successfully
- [ ] Downloaded all required models
- [ ] Installed LanPaint custom node
- [ ] Created virtual environment and installed dependencies
- [ ] Started ComfyUI with desired GPU configuration
- [ ] Verified ComfyUI is accessible (curl test)
- [ ] Tested face swap on 5 rows successfully
- [ ] Started full production run in tmux
- [ ] Monitoring progress regularly

## Support

If you encounter issues:

1. Check `face_swap.log` for detailed error messages
2. Check `processing_log.json` for success/failure stats
3. Check ComfyUI terminal output for workflow errors
4. Verify GPU status with `nvidia-smi`
5. Ensure all models are downloaded correctly

## Estimated Timeline

- **Setup & Testing**: 1-2 hours
- **Full Processing (1128 rows)**: 20-35 hours
- **Total Project**: ~24-36 hours

Run in tmux/screen so you can disconnect and reconnect without interrupting the process.

---

**Ready to start? Follow the steps above sequentially!**
