# Setup Guide for GPU Server

Complete setup instructions for running the face swap automation on your H100 GPU server.

## Prerequisites

- 4x H100 GPUs
- Ubuntu/Linux OS
- Python 3.8+
- CUDA installed
- Git access to https://github.com/ch33nchan/character-swap

## Step 1: Clone Repository

```bash
git clone https://github.com/ch33nchan/character-swap
cd character-swap
```

## Step 2: Run Setup Script

The setup script will:
- Verify Python and CUDA installation
- Check for ComfyUI (install if missing)
- Verify required models
- Test ComfyUI startup

```bash
chmod +x setup_comfyui.sh
./setup_comfyui.sh
```

### Manual ComfyUI Installation (if needed)

If the setup script doesn't install ComfyUI automatically:

```bash
# Install ComfyUI
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install PyTorch with CUDA support
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ComfyUI requirements
pip3 install -r requirements.txt

# Install custom nodes
cd custom_nodes
git clone https://github.com/scraed/LanPaint.git
cd LanPaint
pip3 install -r requirements.txt
```

## Step 3: Download Required Models

### Model Directory Structure

```
~/ComfyUI/models/
├── unet/
│   └── flux-2-klein-9b.safetensors
├── vae/
│   └── flux2-vae.safetensors
├── clip/
│   └── qwen_3_8b_fp8mixed.safetensors
└── loras/
    └── bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors
```

### Download Commands

```bash
cd ~/ComfyUI/models

# Download Flux2 Klein 9b UNET
mkdir -p unet
cd unet
wget https://huggingface.co/Comfy-Org/flux2-klein-9b/resolve/main/flux-2-klein-9b.safetensors

# Download VAE
cd ../
mkdir -p vae
cd vae
wget https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors

# Download CLIP
cd ../
mkdir -p clip
cd clip
wget https://huggingface.co/Comfy-Org/qwen-2.5-1.5b-t2v-text-encoder/resolve/main/qwen_3_8b_fp8mixed.safetensors

# LoRA model - copy from your source
cd ../
mkdir -p loras
# Copy bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors to this directory
```

## Step 4: Setup Project Environment

```bash
cd ~/character-swap

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

## Step 5: Start ComfyUI

### Option A: Use All 4 H100 GPUs

```bash
cd ~/ComfyUI
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --listen 0.0.0.0 --port 8188 --highvram
```

### Option B: Use Specific GPUs

```bash
# Use only GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --listen 0.0.0.0 --port 8188 --highvram
```

### Verify ComfyUI is Running

```bash
# In another terminal
curl http://localhost:8188/system_stats
```

You should see JSON output with system information.

## Step 6: Run Face Swap Pipeline

### Using All 4 GPUs

```bash
cd ~/character-swap
source venv/bin/activate

python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --comfyui-url http://localhost:8188 \
    --gpu-ids 0,1,2,3 \
    --batch-size 10
```

### Test Run (First 5 Rows)

```bash
python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --comfyui-url http://localhost:8188 \
    --gpu-ids 0,1,2,3 \
    --end-row 5 \
    --no-git
```

### Monitor Progress

```bash
# Watch logs in real-time
tail -f face_swap.log

# Check output directory
ls -la data/output/

# View processing summary
cat processing_log.json
```

## Step 7: Long-Running Job Management

For processing all 1128 rows, use `tmux` or `screen`:

```bash
# Start tmux session
tmux new -s faceswap

# Run the pipeline
python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --comfyui-url http://localhost:8188 \
    --gpu-ids 0,1,2,3

# Detach from session: Ctrl+b, then d
# Reattach later: tmux attach -t faceswap
```

## GPU Selection Guide

### Check Available GPUs

```bash
nvidia-smi
```

### GPU ID Assignment

- `--gpu-ids 0,1,2,3` - Use all 4 H100s
- `--gpu-ids 0` - Use only first GPU
- `--gpu-ids 0,1` - Use first two GPUs
- `--gpu-ids 2,3` - Use last two GPUs

### Performance Notes

- **Single H100**: ~2-3 minutes per image
- **4x H100s**: ~1-2 minutes per image (with parallel processing in ComfyUI)
- **Expected total time (1128 rows)**: 20-35 hours with all 4 GPUs

## Troubleshooting

### Issue: ComfyUI Won't Start

```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip3 install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory

```bash
# Start ComfyUI with lower VRAM mode
python3 main.py --listen 0.0.0.0 --port 8188 --normalvram

# Or use fewer GPUs
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --listen 0.0.0.0 --port 8188
```

### Issue: Connection Refused

```bash
# Check if ComfyUI is running
ps aux | grep python | grep main.py

# Check port
netstat -tulpn | grep 8188

# Try local connection
curl http://localhost:8188/system_stats
```

### Issue: Models Not Found

```bash
# Verify model paths
ls -lh ~/ComfyUI/models/unet/
ls -lh ~/ComfyUI/models/vae/
ls -lh ~/ComfyUI/models/clip/
ls -lh ~/ComfyUI/models/loras/

# Check ComfyUI can see models
cd ~/ComfyUI
python3 -c "from folder_paths import get_filename_list; print(get_filename_list('checkpoints'))"
```

## Performance Optimization

### For Maximum Speed

```bash
# Use all GPUs with high VRAM mode
cd ~/ComfyUI
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --listen 0.0.0.0 --port 8188 --highvram

# In face swap script, use aggressive batch size
python3 face_swap.py --batch-size 5 --gpu-ids 0,1,2,3 ...
```

### For Stability

```bash
# Conservative settings
python3 main.py --listen 0.0.0.0 --port 8188 --normalvram
python3 face_swap.py --batch-size 10 --gpu-ids 0,1 ...
```

## Resuming After Interruption

If processing stops:

```bash
# Check last completed row
ls data/output/ | tail -n 1

# Resume from next row
python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --start-row 151 \
    --gpu-ids 0,1,2,3
```

## Next Steps

1. Run setup script: `./setup_comfyui.sh`
2. Download missing models
3. Test ComfyUI: Start server and access http://localhost:8188
4. Test face swap on 5 rows: Use `--end-row 5 --no-git`
5. Run full batch: Remove `--end-row` and `--no-git` flags
6. Monitor progress: `tail -f face_swap.log`

## Support

Check logs for errors:
- `face_swap.log` - Pipeline execution log
- `processing_log.json` - Processing results summary
- ComfyUI terminal output - Workflow execution details
