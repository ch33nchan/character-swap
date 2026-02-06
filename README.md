# Character Swap Automation

Batch face swap using ComfyUI Flux2 Klein 9b + MediaPipe. Goal: same character (including hair, attire) in original pose and background. Output: minimal CSV (row + Swap_Image_Formula); full run details in results.json.

## Scope

- **Current pipeline**: Face swap only (face inpainting). Full character + hair + attire in original pose/background requires a different ComfyUI workflow (e.g. full-body segmentation + inpainting).
- **LoRA training**: Trains on full character (face, hair, attire) from multiple columns: Generated Image, Reference Angle, Front Angle.

## Prerequisites

- Python 3.10+, CUDA GPU, ComfyUI with LanPaint
- Models: flux-2-klein-9b.safetensors, flux2-vae, qwen text encoder, bfs_head_v1 LoRA

## Quick Start

```bash
git clone https://github.com/ch33nchan/character-swap.git
cd character-swap
python3 -m venv venv && source venv/bin/activate
pip3 install -r requirements.txt
```

ComfyUI (separate venv): install ComfyUI, LanPaint, models; then:

```bash
cd ~/ComfyUI && source comfyui_venv/bin/activate
python main.py --listen 0.0.0.0 --port 8189 --highvram
```

## GPU server quick reference

```bash
cd /mnt/data1/srini/character-swap
source venv/bin/activate

# Face swap (ComfyUI must be running on 8189)
python3 face_swap_final.py --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" --quality high --comfyui-url http://localhost:8189 --update-csv --output-csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze_results.csv"

# Upload to Azure and sheet CSV (local or server)
python3 upload_and_format.py --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze_results.csv" --output-csv sheet.csv

# LoRA: setup once, then train
python3 train_character_lora.py setup --work-dir lora_training
export HF_TOKEN=your_token
python3 train_character_lora.py train --use-hf --gpu-id 0 --steps 1500 --work-dir lora_training
```

## Commands to Run

### 1. Face swap (GPU server)

```bash
cd /mnt/data1/srini/character-swap   # or your project path
source venv/bin/activate

python3 face_swap_final.py \
  --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv" \
  --quality high \
  --comfyui-url http://localhost:8189 \
  --update-csv \
  --output-csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze_results.csv"
```

Quality: `fast` (4 steps), `balanced` (8), `high` (12), `ultra` (20). Use `--timeout 900` for ultra. Full run details written to `results.json`.

### 2. Upload to Azure and get sheet CSV (minimal: row + Swap_Image_Formula)

```bash
python3 upload_and_format.py \
  --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze_results.csv" \
  --output-csv "sheet.csv"
```

Produces: `sheet.csv` (columns: row, Swap_Image_Formula) and `upload_results.json` (upload details). Import `sheet.csv` into Google Sheets for =IMAGE() display.

## face_swap_final.py arguments

| Argument | Description |
|----------|-------------|
| `--csv` | Input CSV (required) |
| `--workflow` | Workflow JSON (optional if using --quality) |
| `--quality` | fast \| balanced \| high \| ultra |
| `--comfyui-url` | Default http://localhost:8189 |
| `--start-row`, `--end-row` | Row range (1-indexed) |
| `--update-csv` | Write results into CSV |
| `--output-csv` | Output CSV path |
| `--timeout` | ComfyUI wait seconds (default 600) |

## LoRA training (full character: face, hair, attire)

Trains on FLUX.2-klein-9B. By default uses three CSV columns: **Generated Image**, **Reference Angle**, **Front Angle** so the LoRA learns the full character (hair, attire) from multiple angles.

### Local: prepare dataset

```bash
cd /path/to/character-swap
source venv/bin/activate

python3 train_character_lora.py prepare \
  --csv "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze_results.csv" \
  --max-images 30 \
  --trigger-word "mychar" \
  --character-name "mychar"
```

Optional: `--source-column "Generated Image" "Reference Angle" "Front Angle"` (default). Use a single column with `--source-column "Generated Image"`.

### GPU server: setup and train

```bash
cd /mnt/data1/srini/character-swap
source venv/bin/activate

# 1. Setup AI-Toolkit (once)
python3 train_character_lora.py setup --work-dir lora_training

# 2. Train (HF token required if model is gated)
export HF_TOKEN=your_token
python3 train_character_lora.py train --use-hf --gpu-id 0 --steps 1500 --work-dir lora_training

# 3. Test
python3 train_character_lora.py test --lora-path "lora_training/output/character_expression_lora/character_expression_lora.safetensors" --work-dir lora_training
```

Train args: `--model-path`, `--use-hf`, `--steps`, `--lr`, `--batch-size`, `--lora-rank`. Copy trained LoRA to ComfyUI `models/loras/` and use the trigger word in prompts.

## Output layout

- `results.json` – run_info, summary, per-row results (from face_swap_final.py)
- `upload_results.json` – upload source_csv, output_csv, uploaded_count, details (from upload_and_format.py)
- `sheet.csv` – minimal: row, Swap_Image_Formula only
- `data/output/row_XXXX/result.png` – result images

## Troubleshooting

- ComfyUI: `curl http://localhost:8189/system_stats`
- NumPy conflicts: use separate venvs for ComfyUI and character-swap
- Timeout: increase `--timeout` (e.g. 900 for ultra quality)
- Logs: `face_swap.log`, `results.json`

Repository: https://github.com/ch33nchan/character-swap
