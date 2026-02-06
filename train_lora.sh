#!/bin/bash
# Wrapper script to run LoRA training with correct venv
# Usage: ./train_lora.sh [command] [args]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/lora_training"
VENV_PATH="${WORK_DIR}/ai-toolkit/venv"

# Activate character-swap venv for prepare/setup
if [ "$1" = "prepare" ] || [ "$1" = "setup" ]; then
    if [ -d "${SCRIPT_DIR}/venv" ]; then
        source "${SCRIPT_DIR}/venv/bin/activate"
    fi
    python3 train_character_lora.py "$@"
    exit $?
fi

# For train/test, use ai-toolkit venv
if [ "$1" = "train" ] || [ "$1" = "test" ]; then
    if [ ! -d "$VENV_PATH" ]; then
        echo "ERROR: AI-Toolkit venv not found at: $VENV_PATH"
        echo "Run: ./train_lora.sh setup first"
        exit 1
    fi
    
    # Activate ai-toolkit venv
    source "${VENV_PATH}/bin/activate"
    
    # Install pandas in ai-toolkit venv if needed (for train script imports)
    if ! python -c "import pandas" 2>/dev/null; then
        echo "Installing pandas in ai-toolkit venv..."
        pip install pandas requests pillow
    fi
    
    # Run training
    python3 train_character_lora.py "$@"
    
    deactivate
    exit $?
fi

# Default: show help
echo "LoRA Training Wrapper"
echo ""
echo "Usage:"
echo "  ./train_lora.sh prepare --csv <file> --max-images 30"
echo "  ./train_lora.sh setup"
echo "  ./train_lora.sh train --gpu-id 0 --steps 1500"
echo "  ./train_lora.sh test --lora-path <path>"
echo ""
echo "Commands:"
echo "  prepare  - Prepare training dataset"
echo "  setup    - Setup AI-Toolkit"
echo "  train    - Train LoRA"
echo "  test     - Test trained LoRA"
