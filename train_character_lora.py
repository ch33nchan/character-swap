#!/usr/bin/env python3
"""
Complete LoRA Training Pipeline for Character + Expression Control
Single script to prepare dataset, generate configs, and train on GPU server

Usage:
    # Step 1: Prepare dataset locally
    python3 train_character_lora.py prepare --csv "path/to/csv" --max-images 30
    
    # Step 2: On GPU server, setup and train
    python3 train_character_lora.py setup
    python3 train_character_lora.py train --gpu-id 0
    
    # Step 3: Test trained LoRA
    python3 train_character_lora.py test --lora-path "output/character_lora.safetensors"
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
    
    def __init__(self, work_dir: str = "lora_training"):
        self.work_dir = Path(work_dir)
        self.dataset_dir = self.work_dir / "dataset"
        self.images_dir = self.dataset_dir / "images"
        self.config_dir = self.work_dir / "config"
        self.output_dir = self.work_dir / "output"
        self.toolkit_dir = self.work_dir / "ai-toolkit"
        
        # Character name (customize this)
        self.character_name = "CHARNAME"
        self.trigger_word = "CHARNAME"
    
    def prepare_dataset(self, csv_path: str, max_images: int = 30, 
                       source_column: str = "Generated Image"):
        """
        Step 1: Download images from CSV and create training dataset
        """
        logger.info("="*60)
        logger.info("STEP 1: DATASET PREPARATION")
        logger.info("="*60)
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV: {len(df)} rows")
        
        # Download images
        dataset_info = []
        successful = 0
        
        for idx in range(min(max_images, len(df))):
            row = df.iloc[idx]
            
            try:
                # Get image URL
                image_url = row[source_column]
                if pd.isna(image_url) or not str(image_url).startswith('http'):
                    logger.warning(f"Row {idx+1}: Invalid URL, skipping")
                    continue
                
                # Download image
                logger.info(f"[{idx+1}/{max_images}] Downloading from row {idx+1}...")
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                
                # Save image
                img_path = self.images_dir / f"image_{idx:04d}.png"
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify image
                img = Image.open(img_path)
                width, height = img.size
                logger.info(f"  ✓ Saved: {img_path.name} ({width}x{height})")
                
                # Create caption
                caption = self._create_caption(row, idx)
                caption_path = self.images_dir / f"image_{idx:04d}.txt"
                with open(caption_path, 'w') as f:
                    f.write(caption)
                
                logger.info(f"  ✓ Caption: {caption[:80]}...")
                
                dataset_info.append({
                    'image': str(img_path),
                    'caption': caption,
                    'original_row': idx + 1,
                    'size': f"{width}x{height}"
                })
                
                successful += 1
                
            except Exception as e:
                logger.error(f"Row {idx+1}: Failed - {e}")
                continue
        
        # Save dataset metadata
        metadata = {
            'total_images': successful,
            'character_name': self.character_name,
            'trigger_word': self.trigger_word,
            'source_csv': csv_path,
            'images': dataset_info
        }
        
        metadata_path = self.dataset_dir / 'dataset_info.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Analyze dataset
        self._analyze_dataset()
        
        logger.info("\n" + "="*60)
        logger.info(f"✓ Dataset prepared: {successful} images")
        logger.info(f"✓ Location: {self.images_dir}")
        logger.info(f"✓ Metadata: {metadata_path}")
        logger.info("="*60)
        
        return successful
    
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
        """Extract character-specific features from prompt"""
        if not prompt:
            return "portrait, professional photo"
        
        # Keywords that indicate character features
        feature_keywords = [
            'face', 'hair', 'eyes', 'skin', 'clothing', 'wearing',
            'style', 'look', 'appearance', 'features', 'expression'
        ]
        
        lines = prompt.split('\n')
        feature_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in feature_keywords):
                # Clean up the line
                clean_line = line.strip('- ').strip()
                if clean_line and not clean_line.startswith('CRITICAL'):
                    feature_lines.append(clean_line)
        
        if feature_lines:
            # Combine and simplify
            features = ' '.join(feature_lines[:3])  # Take first 3 relevant lines
            features = re.sub(r'\s+', ' ', features)  # Clean whitespace
            return features
        
        return "portrait, professional photo, high quality"
    
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
        
        # Install additional dependencies
        subprocess.run([
            str(pip_path), 'install',
            'transformers', 'accelerate', 'safetensors', 
            'omegaconf', 'pyyaml'
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
        
        # Determine model path
        if use_local_model and model_path is None:
            # Try to find local ComfyUI models
            comfyui_model = Path.home() / "ComfyUI" / "models" / "unet" / "flux-2-klein-9b.safetensors"
            if comfyui_model.exists():
                model_path = str(comfyui_model.parent.parent)  # Point to ComfyUI/models/
                logger.info(f"Using local model: {model_path}")
            else:
                # Fallback to HuggingFace (requires HF_TOKEN)
                model_path = "black-forest-labs/FLUX.2-dev"
                logger.warning("Local model not found, using HuggingFace (requires HF_TOKEN)")
        elif model_path is None:
            model_path = "black-forest-labs/FLUX.2-dev"
        
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
    
    def train(self, gpu_id: int = 0, config_path: Optional[str] = None):
        """
        Step 4: Execute training
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
        
        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Using GPU: {gpu_id}")
        
        # Run training
        # Use sys.executable to get the currently active Python (from venv)
        python_path = Path(sys.executable)
        run_script = self.toolkit_dir / 'run.py'
        
        # Convert to absolute paths
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
            ], check=True, cwd=str(self.toolkit_dir))
            
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
        
        logger.info("Loading Flux pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
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
    prepare_parser = subparsers.add_parser('prepare', help='Prepare training dataset')
    prepare_parser.add_argument('--csv', required=True, help='Path to CSV file')
    prepare_parser.add_argument('--max-images', type=int, default=30, help='Maximum images to download')
    prepare_parser.add_argument('--source-column', default='Generated Image', help='CSV column with image URLs')
    prepare_parser.add_argument('--work-dir', default='lora_training', help='Working directory')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup AI-Toolkit')
    setup_parser.add_argument('--work-dir', default='lora_training', help='Working directory')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train LoRA')
    train_parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use')
    train_parser.add_argument('--steps', type=int, default=1500, help='Training steps')
    train_parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    train_parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    train_parser.add_argument('--model-path', default=None, help='Path to base model (default: auto-detect ComfyUI models)')
    train_parser.add_argument('--use-hf', action='store_true', help='Use HuggingFace model instead of local')
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
    
    # Initialize pipeline
    pipeline = LoRATrainingPipeline(work_dir=args.work_dir)
    
    # Execute command
    if args.command == 'prepare':
        pipeline.prepare_dataset(
            csv_path=args.csv,
            max_images=args.max_images,
            source_column=args.source_column
        )
        # Also create config
        pipeline.create_training_config()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Transfer dataset to GPU server:")
        print(f"   scp -r {args.work_dir} user@gpu-server:~/")
        print("")
        print("2. On GPU server, setup AI-Toolkit:")
        print(f"   python3 train_character_lora.py setup --work-dir {args.work_dir}")
        print("")
        print("3. Train LoRA:")
        print(f"   python3 train_character_lora.py train --gpu-id 0 --work-dir {args.work_dir}")
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
        
        success = pipeline.train(gpu_id=args.gpu_id, config_path=str(config_path))
        
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
