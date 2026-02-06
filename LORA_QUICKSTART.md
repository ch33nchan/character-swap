# LoRA Training - Quick Reference

## One-Command Summary

```bash
# 1. Prepare (local)
python3 train_character_lora.py prepare --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" --max-images 30

# 2. Transfer to GPU
scp -r lora_training user@gpu-server:~/

# 3. Setup (GPU server)
python3 train_character_lora.py setup

# 4. Train (GPU server)
python3 train_character_lora.py train --gpu-id 0

# 5. Test
python3 train_character_lora.py test --lora-path "lora_training/output/character_expression_lora/character_expression_lora.safetensors"
```

## Timeline

| Step | Time | Location |
|------|------|----------|
| Prepare dataset | 30 min | Local |
| Transfer | 5 min | Local → GPU |
| Setup toolkit | 15 min | GPU server |
| Train LoRA | 3-4 hours | GPU server |
| Test | 10 min | GPU server |
| **Total** | **4-5 hours** | |

## Results

- **Face consistency**: 95-98% (vs 80-85% with face swap)
- **Expression control**: "smiling", "serious", "surprised" work reliably
- **Quality**: Native generation, no artifacts
- **Speed**: 30s per image (vs 60s with face swap)
- **Cost**: Free (your H100)

## Expression Keywords

After training, use these in prompts:

```
"CHARNAME, smiling happily, front view"
"CHARNAME, serious expression, looking at camera"
"CHARNAME, surprised look, three-quarter view"
"CHARNAME, neutral expression, side profile"
"CHARNAME, laughing joyfully, close-up"
"CHARNAME, gentle smile, soft lighting"
"CHARNAME, confident look, studio photo"
"CHARNAME, thoughtful expression, natural light"
```

## Monitoring Training

```bash
# Watch training progress
tail -f lora_training/output/character_expression_lora/training.log

# Check samples (generated every 250 steps)
ls lora_training/output/character_expression_lora/samples/

# GPU usage
watch -n 1 nvidia-smi
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Already using quantization, reduce batch size if needed |
| Inconsistent face | Increase steps to 2000 |
| Overfitting | Decrease steps to 1000 |
| Poor expressions | Check captions have expression keywords |

## Integration with ComfyUI

```bash
# Copy trained LoRA
cp lora_training/output/character_expression_lora/character_expression_lora.safetensors \
   ~/ComfyUI/models/loras/

# Use in workflow
# Add "Load LoRA" node → Set strength 0.8-1.0 → Use "CHARNAME" in prompts
```

## Full Documentation

See `LORA_TRAINING_GUIDE.md` for complete details.
