#!/usr/bin/env python3
"""
LoRA training for full character (face, hair, attire) + expression.
Uses multiple image columns by default: Generated Image, Reference Angle, Front Angle.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoRATrainingPipeline:
    """Complete pipeline for training character expression LoRA"""
    
    def __init__(self, work_dir: str = "lora_training", character_name: str = "CHARNAME", trigger_word: Optional[str] = None):
        self.work_dir = Path(work_dir)
        self.dataset_dir = self.work_dir / "dataset"
        self.images_dir = self.dataset_dir / "images"
        self.config_dir = self.work_dir / "config"
        self.output_dir = self.work_dir / "output"
        self.toolkit_dir = self.work_dir / "ai-toolkit"
        self.character_name = character_name
        self.trigger_word = trigger_word or character_name

    def prepare_dataset(self, csv_path: str, max_images: int = 30,
                       source_columns: Optional[List[str]] = None) -> int:
        """
        Download images from one or more CSV columns (full character: hair, attire, multiple angles).
        Default columns: Generated Image, Reference Angle, Front Angle.
        """
        logger.info("="*60)
        logger.info("STEP 1: DATASET PREPARATION (full character: hair, attire)")
        logger.info("="*60)
        if source_columns is None:
            source_columns = ["Generated Image", "Reference Angle", "Front Angle"]
        self.images_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV: {len(df)} rows, columns: {source_columns}")
        for col in source_columns:
            if col not in df.columns:
                logger.warning(f"Column not in CSV: {col}, skipping")
        source_columns = [c for c in source_columns if c in df.columns]
        if not source_columns:
            raise ValueError("No valid source columns found in CSV")
        dataset_info = []
        global_idx = 0
        max_rows = min(max_images, len(df))
        for row_idx in range(max_rows):
            row = df.iloc[row_idx]
            caption = self._create_caption(row, row_idx)
            for col in source_columns:
                url = row.get(col)
                if pd.isna(url) or not str(url).strip().startswith("http"):
                    continue
                try:
                    logger.info(f"[{global_idx+1}] Row {row_idx+1} ({col})...")
                    response = requests.get(str(url).strip(), timeout=30)
                    response.raise_for_status()
                    img_path = self.images_dir / f"image_{global_idx:04d}.png"
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    img = Image.open(img_path)
                    w, h = img.size
                    cap_path = self.images_dir / f"image_{global_idx:04d}.txt"
                    with open(cap_path, "w") as f:
                        f.write(caption)
                    logger.info(f"  Saved {img_path.name} ({w}x{h}), caption: {caption[:60]}...")
                    dataset_info.append({
                        "image": str(img_path),
                        "caption": caption,
                        "original_row": row_idx + 1,
                        "column": col,
                        "size": f"{w}x{h}",
                    })
                    global_idx += 1
                except Exception as e:
                    logger.error(f"Row {row_idx+1} {col}: {e}")
        metadata = {
            "total_images": global_idx,
            "character_name": self.character_name,
            "trigger_word": self.trigger_word,
            "source_csv": csv_path,
            "source_columns": source_columns,
            "images": dataset_info,
        }
        metadata_path = self.dataset_dir / "dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self._analyze_dataset()
        logger.info("\n" + "="*60)
        logger.info(f"Dataset prepared: {global_idx} images (full character: hair, attire)")
        logger.info(f"Location: {self.images_dir}, metadata: {metadata_path}")
        logger.info("="*60)
        return global_idx
    
    def _create_caption(self, row: pd.Series, idx: int) -> str:
        """
        Create training caption from CSV row
        Format: "CHARNAME, [expression], [description]"
        """
        # Try to extract character description from Edit Prompt
        prompt = row.get('Edit Prompt', '')
        
        # Extract character features (face, hair, clothing, etc.)
        char_features = self._extract_character_features(prompt)
        
        # Detect expression if available
        expression = self._detect_expression(prompt, idx)
        
        # Build caption
        if expression:
            caption = f"{self.trigger_word}, {expression}, {char_features}"
        else:
            caption = f"{self.trigger_word}, {char_features}"
        
        # Clean up caption
        caption = ' '.join(caption.split())  # Remove extra whitespace
        caption = caption[:500]  # Limit length
        
        return caption
    
    def _extract_character_features(self, prompt: str) -> str:
        """Extract full-character features: face, hair, attire, body."""
        if not prompt:
            return "full body, portrait, professional photo, high quality"
        feature_keywords = [
            "face", "hair", "eyes", "skin", "clothing", "wearing", "attire", "outfit",
            "style", "look", "appearance", "features", "expression", "body", "pose",
        ]
        lines = prompt.split("\n")
        feature_lines = []
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in feature_keywords):
                clean = line.strip("- ").strip()
                if clean and not clean.startswith("CRITICAL"):
                    feature_lines.append(clean)
        if feature_lines:
            features = " ".join(feature_lines[:5])
            return re.sub(r"\s+", " ", features)
        return "full body, portrait, professional photo, hair and attire, high quality"
    
    def _detect_expression(self, prompt: str, idx: int) -> Optional[str]:
        """Detect or assign expression based on prompt or index"""
        prompt_lower = prompt.lower() if prompt else ""
        
        # Expression keywords
        expressions = {
            'smiling': ['smile', 'smiling', 'happy', 'cheerful', 'grin'],
            'serious': ['serious', 'stern', 'focused', 'intense'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished'],
            'neutral': ['neutral', 'calm', 'composed'],
            'laughing': ['laugh', 'laughing', 'joyful'],
            'thoughtful': ['thinking', 'thoughtful', 'pensive']
        }
        
        # Check for expression keywords in prompt
        for expr, keywords in expressions.items():
            if any(kw in prompt_lower for kw in keywords):
                return expr
        
        # Assign expressions in rotation for diversity
        expr_list = ['neutral expression', 'slight smile', 'serious expression', 
                     'gentle smile', 'confident look']
        return expr_list[idx % len(expr_list)]
    
    def _analyze_dataset(self):
        """Analyze dataset for diversity and quality"""
        logger.info("\nAnalyzing dataset diversity...")
        
        images = list(self.images_dir.glob('*.png'))
        captions = list(self.images_dir.glob('*.txt'))
        
        logger.info(f"  Images: {len(images)}")
        logger.info(f"  Captions: {len(captions)}")
        
        # Analyze image sizes
        sizes = []
        for img_path in images:
            img = Image.open(img_path)
            sizes.append(img.size)
        
        if sizes:
            avg_width = sum(s[0] for s in sizes) / len(sizes)
            avg_height = sum(s[1] for s in sizes) / len(sizes)
            logger.info(f"  Average size: {avg_width:.0f}x{avg_height:.0f}")
        
        # Analyze captions
        expressions = {}
        for cap_path in captions:
            with open(cap_path) as f:
                caption = f.read()
                # Count expression keywords
                for expr in ['smiling', 'serious', 'neutral', 'surprised', 'laughing']:
                    if expr in caption.lower():
                        expressions[expr] = expressions.get(expr, 0) + 1
        
        if expressions:
            logger.info(f"  Expression distribution:")
            for expr, count in sorted(expressions.items()):
                logger.info(f"    {expr}: {count}")
    
    def setup_toolkit(self):
        """
        Step 2: Setup AI-Toolkit on GPU server
        """
        logger.info("="*60)
        logger.info("STEP 2: AI-TOOLKIT SETUP")
        logger.info("="*60)
        
        # Clone AI-Toolkit
        if not self.toolkit_dir.exists():
            logger.info("Cloning AI-Toolkit...")
            subprocess.run([
                'git', 'clone', 
                'https://github.com/ostris/ai-toolkit.git',
                str(self.toolkit_dir)
            ], check=True)
            logger.info("✓ AI-Toolkit cloned")
        else:
            logger.info("✓ AI-Toolkit already exists")
        
        # Create venv
        venv_dir = self.toolkit_dir / 'venv'
        if not venv_dir.exists():
            logger.info("Creating virtual environment...")
            subprocess.run([
                sys.executable, '-m', 'venv', str(venv_dir)
            ], check=True)
            logger.info("✓ Virtual environment created")
        
        # Install dependencies
        logger.info("Installing dependencies...")
        pip_path = venv_dir / 'bin' / 'pip'
        
        # Install PyTorch with CUDA
        subprocess.run([
            str(pip_path), 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu121'
        ], check=True)
        
        # Install AI-Toolkit requirements
        requirements = self.toolkit_dir / 'requirements.txt'
        if requirements.exists():
            subprocess.run([
                str(pip_path), 'install', '-r', str(requirements)
            ], check=True)
        
        subprocess.run([
            str(pip_path), 'install',
            'transformers', 'accelerate', 'safetensors',
            'omegaconf', 'pyyaml', 'oyaml'
        ], check=True)
        logger.info("✓ Dependencies installed")
        logger.info("="*60)
    
    def create_training_config(self, steps: int = 1500, learning_rate: float = 0.0004,
                              batch_size: int = 1, lora_rank: int = 16, 
                              model_path: str = None, use_local_model: bool = True):
        """
        Step 3: Create training configuration
        """
        logger.info("="*60)
        logger.info("STEP 3: TRAINING CONFIGURATION")
        logger.info("="*60)
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine model path (default: FLUX.2-klein-9B for character swap alignment)
        default_hf_model = "black-forest-labs/FLUX.2-klein-9B"
        if use_local_model and model_path is None:
            comfyui_model = Path.home() / "ComfyUI" / "models" / "unet" / "flux-2-klein-9b.safetensors"
            if comfyui_model.exists():
                model_path = str(comfyui_model.parent.parent)
                logger.info(f"Using local model: {model_path}")
            else:
                model_path = default_hf_model
                logger.warning("Local model not found, using HuggingFace (requires HF_TOKEN for gated model)")
        elif model_path is None:
            model_path = default_hf_model
        
        # Sample prompts for validation during training
        sample_prompts = [
            f"{self.trigger_word}, smiling happily, front view, professional portrait",
            f"{self.trigger_word}, serious expression, looking at camera, studio lighting",
            f"{self.trigger_word}, surprised look, three-quarter view, natural light",
            f"{self.trigger_word}, neutral expression, side profile, high quality photo",
            f"{self.trigger_word}, gentle smile, close-up portrait, soft lighting"
        ]
        
        config = {
            'job': 'extension',
            'config': {
                'name': 'character_expression_lora',
                'process': [{
                    'type': 'sd_trainer',
                    'training_folder': str(self.output_dir),
                    'device': 'cuda:0',
                    
                    'model': {
                        'name_or_path': model_path,
                        'is_flux': True,
                        'quantize': True
                    },
                    
                    'network': {
                        'type': 'lora',
                        'linear': lora_rank,
                        'linear_alpha': lora_rank
                    },
                    
                    'datasets': [{
                        'folder_path': str(self.images_dir),
                        'caption_ext': 'txt',
                        'caption_dropout_rate': 0.05,
                        'shuffle_tokens': False,
                        'cache_latents_to_disk': True,
                        'resolution': [1024, 1024]
                    }],
                    
                    'train': {
                        'batch_size': batch_size,
                        'steps': steps,
                        'gradient_accumulation_steps': 4,
                        'train_unet': True,
                        'train_text_encoder': False,
                        'lr': learning_rate,
                        'optimizer': 'adamw8bit',
                        'lr_scheduler': 'constant',
                        'save_every': 250,
                        'sample_every': 250,
                        'sample_prompts': sample_prompts
                    },
                    
                    'sample': {
                        'sampler': 'flowmatch',
                        'sample_steps': 20,
                        'cfg_scale': 3.5,
                        'height': 1024,
                        'width': 1024
                    }
                }]
            }
        }
        
        # Save config as YAML
        config_path = self.config_dir / 'character_lora.yaml'
        
        # Convert to YAML format manually (simple approach)
        yaml_content = self._dict_to_yaml(config)
        
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"✓ Config created: {config_path}")
        logger.info(f"  Training steps: {steps}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  LoRA rank: {lora_rank}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info("="*60)
        
        return config_path
    
    def _dict_to_yaml(self, data: dict, indent: int = 0) -> str:
        """Simple dict to YAML converter"""
        lines = []
        prefix = '  ' * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                for line in self._dict_to_yaml(value, indent + 1).split('\n'):
                    if line.strip():
                        lines.append(line)
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        for line in self._dict_to_yaml(item, indent + 2).split('\n'):
                            if line.strip():
                                lines.append(f"  {line}")
                    else:
                        lines.append(f"{prefix}  - {self._yaml_value(item)}")
            else:
                lines.append(f"{prefix}{key}: {self._yaml_value(value)}")
        
        return '\n'.join(lines)
    
    def _yaml_value(self, value):
        """Format value for YAML"""
        if isinstance(value, bool):
            return 'true' if value else 'false'
        elif isinstance(value, str):
            if ' ' in value or ':' in value:
                return f'"{value}"'
            return value
        return str(value)
    
    def train(self, gpu_id: int = 0, gpu_ids: Optional[str] = None, config_path: Optional[str] = None):
        """
        Step 4: Execute training. Uses AI-Toolkit venv Python (has oyaml). Multi-GPU: set gpu_ids e.g. "0,1,2,3".
        """
        logger.info("="*60)
        logger.info("STEP 4: TRAINING")
        logger.info("="*60)
        if config_path is None:
            config_path = self.config_dir / 'character_lora.yaml'
        if not Path(config_path).exists():
            logger.error(f"Config not found: {config_path}")
            logger.info("Run: python3 train_character_lora.py prepare first")
            return False
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            logger.info(f"Using GPUs: {gpu_ids}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Using GPU: {gpu_id}")
        toolkit_dir_abs = self.toolkit_dir.resolve()
        toolkit_venv_python = toolkit_dir_abs / "venv" / "bin" / "python3"
        if toolkit_venv_python.exists():
            python_path = toolkit_venv_python
        else:
            logger.warning("AI-Toolkit venv not found, using current Python (ensure oyaml is installed: pip install oyaml)")
            python_path = Path(sys.executable).resolve()
        run_script = toolkit_dir_abs / "run.py"
        run_script = run_script.resolve()
        config_path = Path(config_path).resolve()
        
        # Verify files exist
        if not python_path.exists():
            logger.error(f"Python not found: {python_path}")
            logger.error("Make sure you ran: python3 train_character_lora.py setup")
            return False
        
        if not run_script.exists():
            logger.error(f"run.py not found: {run_script}")
            logger.error("AI-Toolkit may not be properly installed")
            return False
        
        logger.info(f"Starting training...")
        logger.info(f"Python: {python_path}")
        logger.info(f"Script: {run_script}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("\nTraining will take 2-4 hours on H100...")
        logger.info("Monitor progress in: {}/character_expression_lora/".format(self.output_dir))
        
        try:
            subprocess.run([
                str(python_path),
                str(run_script),
                str(config_path)
            ], check=True, cwd=str(toolkit_dir_abs))
            
            logger.info("="*60)
            logger.info("✓ Training complete!")
            logger.info("="*60)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def test_lora(self, lora_path: str, output_dir: str = "test_outputs"):
        """
        Step 5: Test trained LoRA
        """
        logger.info("="*60)
        logger.info("STEP 5: TESTING LORA")
        logger.info("="*60)
        
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError:
            logger.error("diffusers not installed. Install with: pip install diffusers")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading Flux pipeline (FLUX.2-klein-9B)...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.bfloat16
        )
        
        logger.info(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)
        pipe.to("cuda")
        
        # Test prompts
        test_prompts = [
            (f"{self.trigger_word}, smiling happily, front view, professional portrait", "01_smiling_front"),
            (f"{self.trigger_word}, serious expression, looking at camera", "02_serious_front"),
            (f"{self.trigger_word}, surprised look, three-quarter view", "03_surprised_3quarter"),
            (f"{self.trigger_word}, neutral expression, side profile", "04_neutral_profile"),
            (f"{self.trigger_word}, laughing joyfully, close-up portrait", "05_laughing_closeup"),
            (f"{self.trigger_word}, gentle smile, soft lighting", "06_gentle_smile"),
            (f"{self.trigger_word}, confident look, studio lighting", "07_confident"),
            (f"{self.trigger_word}, thoughtful expression, natural light", "08_thoughtful")
        ]
        
        logger.info(f"\nGenerating {len(test_prompts)} test images...")
        
        for prompt, filename in test_prompts:
            logger.info(f"\n  Generating: {filename}")
            logger.info(f"  Prompt: {prompt}")
            
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=3.5,
                height=1024,
                width=1024
            ).images[0]
            
            save_path = output_path / f"{filename}.png"
            image.save(save_path)
            logger.info(f"  ✓ Saved: {save_path}")
        
        logger.info("\n" + "="*60)
        logger.info(f"✓ Test complete! Check outputs in: {output_path}")
        logger.info("="*60)
        logger.info("\nEvaluate results for:")
        logger.info("  1. Face consistency - Same person across all images?")
        logger.info("  2. Expression control - Do keywords trigger correct emotions?")
        logger.info("  3. Quality - Sharp, detailed, no artifacts?")
        logger.info("  4. Generalization - Works with new poses/angles?")


def main():
    parser = argparse.ArgumentParser(
        description='Complete LoRA Training Pipeline for Character + Expression Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset from CSV
  python3 train_character_lora.py prepare --csv "data.csv" --max-images 30
  
  # Setup AI-Toolkit on GPU server
  python3 train_character_lora.py setup
  
  # Train LoRA
  python3 train_character_lora.py train --gpu-id 0 --steps 1500
  
  # Test trained LoRA
  python3 train_character_lora.py test --lora-path "lora_training/output/character_expression_lora/character_expression_lora.safetensors"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset (full character: hair, attire)')
    prepare_parser.add_argument('--csv', required=True, help='Path to CSV file')
    prepare_parser.add_argument('--max-images', type=int, default=30, help='Max rows to scan (multiple images per row if multiple columns)')
    prepare_parser.add_argument('--source-column', nargs='*', default=None,
                               help='CSV columns for images (default: Generated Image, Reference Angle, Front Angle)')
    prepare_parser.add_argument('--trigger-word', default='CHARNAME', help='Trigger word for LoRA')
    prepare_parser.add_argument('--character-name', default='CHARNAME', help='Character name (used in captions)')
    prepare_parser.add_argument('--work-dir', default='lora_training', help='Working directory')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup AI-Toolkit')
    setup_parser.add_argument('--work-dir', default='lora_training', help='Working directory')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train LoRA')
    train_parser.add_argument('--gpu-id', type=int, default=0, help='Single GPU ID (ignored if --gpu-ids set)')
    train_parser.add_argument('--gpu-ids', type=str, default=None, help='Multiple GPUs e.g. "0,1,2,3"')
    train_parser.add_argument('--steps', type=int, default=1500, help='Training steps')
    train_parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    train_parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    train_parser.add_argument('--model-path', default=None, help='Base model path or HF id (default: FLUX.2-klein-9B when using HF)')
    train_parser.add_argument('--use-hf', action='store_true', help='Use HuggingFace black-forest-labs/FLUX.2-klein-9B (requires HF token if gated)')
    train_parser.add_argument('--work-dir', default='lora_training', help='Working directory')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test trained LoRA')
    test_parser.add_argument('--lora-path', required=True, help='Path to trained LoRA .safetensors file')
    test_parser.add_argument('--output-dir', default='test_outputs', help='Output directory for test images')
    test_parser.add_argument('--work-dir', default='lora_training', help='Working directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    work_dir = getattr(args, "work_dir", "lora_training")
    character_name = getattr(args, "character_name", "CHARNAME")
    trigger_word = getattr(args, "trigger_word", "CHARNAME")
    pipeline = LoRATrainingPipeline(work_dir=work_dir, character_name=character_name, trigger_word=trigger_word)

    if args.command == "prepare":
        source_columns = getattr(args, "source_column", None)
        if source_columns is not None and len(source_columns) == 0:
            source_columns = None
        pipeline.prepare_dataset(
            csv_path=args.csv,
            max_images=args.max_images,
            source_columns=source_columns,
        )
        pipeline.create_training_config()
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Transfer dataset to GPU server:")
        print(f"   scp -r {work_dir} user@gpu-server:~/")
        print("")
        print("2. On GPU server, setup AI-Toolkit:")
        print(f"   python3 train_character_lora.py setup --work-dir {work_dir}")
        print("")
        print("3. Train LoRA (full character including hair, attire):")
        print(f"   python3 train_character_lora.py train --gpu-id 0 --work-dir {work_dir}")
        print("="*60)
    
    elif args.command == 'setup':
        pipeline.setup_toolkit()
        
        print("\n" + "="*60)
        print("NEXT STEP:")
        print("="*60)
        print("Train LoRA:")
        print(f"  python3 train_character_lora.py train --gpu-id 0 --work-dir {args.work_dir}")
        print("="*60)
    
    elif args.command == 'train':
        # Create config if it doesn't exist
        config_path = pipeline.config_dir / 'character_lora.yaml'
        if not config_path.exists():
            pipeline.create_training_config(
                steps=args.steps,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                lora_rank=args.lora_rank,
                model_path=args.model_path,
                use_local_model=not args.use_hf
            )
        
        success = pipeline.train(gpu_id=args.gpu_id, gpu_ids=args.gpu_ids, config_path=str(config_path))
        
        if success:
            print("\n" + "="*60)
            print("NEXT STEP:")
            print("="*60)
            print("Test trained LoRA:")
            lora_file = pipeline.output_dir / "character_expression_lora" / "character_expression_lora.safetensors"
            print(f"  python3 train_character_lora.py test --lora-path {lora_file}")
            print("="*60)
    
    elif args.command == 'test':
        pipeline.test_lora(
            lora_path=args.lora_path,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
