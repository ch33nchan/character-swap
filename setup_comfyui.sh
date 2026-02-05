#!/bin/bash
set -e

echo "=========================================="
echo "ComfyUI Setup and Verification Script"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python version
echo ""
echo "Step 1: Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Found $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check CUDA/GPU
echo ""
echo "Step 2: Checking CUDA/GPU setup..."
if command -v nvidia-smi &> /dev/null; then
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo ""
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    print_success "Found $GPU_COUNT GPU(s)"
else
    print_error "nvidia-smi not found. CUDA may not be installed."
    exit 1
fi

# Check if ComfyUI is installed
echo ""
echo "Step 3: Checking for ComfyUI installation..."
COMFYUI_DIRS=(
    "$HOME/ComfyUI"
    "$HOME/comfyui"
    "/opt/ComfyUI"
    "/workspace/ComfyUI"
    "./ComfyUI"
)

COMFYUI_PATH=""
for dir in "${COMFYUI_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/main.py" ]; then
        COMFYUI_PATH="$dir"
        print_success "Found ComfyUI at: $COMFYUI_PATH"
        break
    fi
done

if [ -z "$COMFYUI_PATH" ]; then
    print_info "ComfyUI not found in standard locations."
    echo ""
    read -p "Enter ComfyUI installation path (or press Enter to install): " USER_PATH
    
    if [ -n "$USER_PATH" ] && [ -d "$USER_PATH" ]; then
        COMFYUI_PATH="$USER_PATH"
        print_success "Using ComfyUI at: $COMFYUI_PATH"
    else
        echo ""
        print_info "Installing ComfyUI..."
        
        # Install ComfyUI
        INSTALL_DIR="$HOME/ComfyUI"
        git clone https://github.com/comfyanonymous/ComfyUI.git "$INSTALL_DIR"
        cd "$INSTALL_DIR"
        
        # Install requirements
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        python3 -m pip install -r requirements.txt
        
        COMFYUI_PATH="$INSTALL_DIR"
        print_success "ComfyUI installed at: $COMFYUI_PATH"
    fi
fi

# Check required models
echo ""
echo "Step 4: Checking required models..."
MODELS_DIR="$COMFYUI_PATH/models"

check_model() {
    local model_type=$1
    local model_name=$2
    local model_path="$MODELS_DIR/$model_type/$model_name"
    
    if [ -f "$model_path" ]; then
        print_success "Found: $model_type/$model_name"
        return 0
    else
        print_error "Missing: $model_type/$model_name"
        return 1
    fi
}

MISSING_MODELS=0

# Check UNET model
if ! check_model "unet" "flux-2-klein-9b.safetensors"; then
    echo "  Download from: https://huggingface.co/Comfy-Org/flux2-klein-9b"
    ((MISSING_MODELS++))
fi

# Check VAE model
if ! check_model "vae" "flux2-vae.safetensors"; then
    echo "  Download from: https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"
    ((MISSING_MODELS++))
fi

# Check CLIP model
if ! check_model "clip" "qwen_3_8b_fp8mixed.safetensors"; then
    echo "  Download from: https://huggingface.co/Comfy-Org/qwen-2.5-1.5b-t2v-text-encoder"
    ((MISSING_MODELS++))
fi

# Check LoRA model
if ! check_model "loras" "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"; then
    echo "  This is a custom LoRA - ensure you have it from your source"
    ((MISSING_MODELS++))
fi

# Check custom nodes
echo ""
echo "Step 5: Checking custom nodes..."
CUSTOM_NODES_DIR="$COMFYUI_PATH/custom_nodes"

if [ -d "$CUSTOM_NODES_DIR/lanpaint" ] || [ -d "$CUSTOM_NODES_DIR/LanPaint" ]; then
    print_success "Found LanPaint custom node"
else
    print_error "Missing LanPaint custom node"
    echo "  Install with: cd $CUSTOM_NODES_DIR && git clone https://github.com/scraed/LanPaint.git"
    ((MISSING_MODELS++))
fi

if [ -d "$CUSTOM_NODES_DIR/rgthree-comfy" ]; then
    print_success "Found rgthree-comfy custom node"
else
    print_info "rgthree-comfy not found (optional for Image Comparer)"
fi

# Test ComfyUI startup
echo ""
echo "Step 6: Testing ComfyUI startup..."
cd "$COMFYUI_PATH"

# Create a test script to check if ComfyUI can start
cat > /tmp/test_comfyui.py << 'EOF'
import sys
sys.path.insert(0, '.')
try:
    import main
    print("SUCCESS: ComfyUI can be imported")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
EOF

if python3 /tmp/test_comfyui.py; then
    print_success "ComfyUI can start successfully"
else
    print_error "ComfyUI import failed. Check dependencies."
fi

rm /tmp/test_comfyui.py

# Summary
echo ""
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo "ComfyUI Path: $COMFYUI_PATH"
echo "GPU Count: $GPU_COUNT"
echo "Missing Models: $MISSING_MODELS"
echo ""

if [ $MISSING_MODELS -eq 0 ]; then
    print_success "All requirements met! Ready to run."
    echo ""
    echo "To start ComfyUI:"
    echo "  cd $COMFYUI_PATH"
    echo "  python3 main.py --listen 0.0.0.0 --port 8188"
    echo ""
    echo "To use specific GPUs:"
    echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --listen 0.0.0.0 --port 8188"
else
    print_info "Please download missing models before running."
    echo ""
    echo "Models directory: $MODELS_DIR"
fi

echo ""
echo "Next: Run the face swap script with:"
echo "  python3 face_swap.py --csv <csv_file> --workflow <workflow_json> --gpu-ids 0,1,2,3"
