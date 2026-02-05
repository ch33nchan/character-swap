# Quick Start - TL;DR

For full details, see [NEXT_STEPS.md](NEXT_STEPS.md)

## 1. Clone & Setup (5 minutes)

```bash
# On your GPU server
git clone https://github.com/ch33nchan/character-swap.git
cd character-swap
./setup_comfyui.sh
```

## 2. Install Dependencies (2 minutes)

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## 3. Start ComfyUI with 4 H100s

```bash
tmux new -s comfyui
cd ~/ComfyUI
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --listen 0.0.0.0 --port 8188 --highvram
# Ctrl+b, d to detach
```

## 4. Test (5 rows, ~10 minutes)

```bash
cd ~/character-swap
source venv/bin/activate

python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --gpu-ids 0,1,2,3 \
    --end-row 5 \
    --no-git
```

## 5. Run Full Batch (1128 rows, ~20-35 hours)

```bash
tmux new -s faceswap

python3 face_swap.py \
    --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
    --workflow "Flux2 Klein 9b Face Swap.json" \
    --gpu-ids 0,1,2,3
# Ctrl+b, d to detach
```

## 6. Monitor Progress

```bash
# Watch logs
tail -f face_swap.log

# Reattach to session
tmux attach -t faceswap

# Check completed
ls data/output/ | wc -l

# View stats
cat processing_log.json | jq '.summary'
```

## Common Commands

### Check GPU Status
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Verify ComfyUI Running
```bash
curl http://localhost:8188/system_stats
```

### Resume from Row 150
```bash
python3 face_swap.py --start-row 150 --gpu-ids 0,1,2,3 --csv "..." --workflow "..."
```

### Stop Everything
```bash
tmux kill-session -t faceswap
tmux kill-session -t comfyui
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Check ComfyUI running: `ps aux \| grep main.py` |
| Out of memory | Use `--normalvram` instead of `--highvram` |
| Models not found | Run `./setup_comfyui.sh` again |
| Git push failed | Add `--no-git` flag |

## File Locations

- **Code**: `~/character-swap/`
- **ComfyUI**: `~/ComfyUI/`
- **Models**: `~/ComfyUI/models/`
- **Results**: `~/character-swap/data/output/`
- **Logs**: `~/character-swap/face_swap.log`

## Expected Timeline

- Setup: 1-2 hours
- Processing: 20-35 hours (with 4 H100s)
- **Total**: ~24-36 hours

---

**Need help?** See [NEXT_STEPS.md](NEXT_STEPS.md) for detailed instructions.
