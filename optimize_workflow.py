#!/usr/bin/env python3
"""
Optimize Flux2 Klein face swap workflow for better quality
Adjusts key parameters: steps, denoise, CFG, LoRA strength, resolution
"""

import json
import argparse
from pathlib import Path

def optimize_workflow(workflow_path: str, output_path: str = None, 
                     quality_preset: str = "high"):
    """
    Optimize workflow parameters for better face swap quality
    
    Quality presets:
    - fast: 4 steps, denoise 0.8, CFG 1 (current)
    - balanced: 8 steps, denoise 0.9, CFG 2
    - high: 12 steps, denoise 0.95, CFG 3
    - ultra: 20 steps, denoise 1.0, CFG 4
    """
    
    with open(workflow_path) as f:
        workflow = json.load(f)
    
    presets = {
        'fast': {'steps': 4, 'denoise': 0.8, 'cfg': 1, 'lora_strength': 1.0, 'megapixels': 2},
        'balanced': {'steps': 8, 'denoise': 0.9, 'cfg': 2, 'lora_strength': 0.95, 'megapixels': 4},
        'high': {'steps': 12, 'denoise': 0.95, 'cfg': 3, 'lora_strength': 0.9, 'megapixels': 6},
        'ultra': {'steps': 20, 'denoise': 1.0, 'cfg': 4, 'lora_strength': 0.85, 'megapixels': 8.3}
    }
    
    params = presets.get(quality_preset, presets['high'])
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZING WORKFLOW: {quality_preset.upper()} QUALITY")
    print(f"{'='*60}")
    
    # Node 156: LanPaint_KSampler - Main generation parameters
    if "156" in workflow:
        workflow["156"]["inputs"]["steps"] = params['steps']
        workflow["156"]["inputs"]["denoise"] = params['denoise']
        print(f"✓ KSampler: steps={params['steps']}, denoise={params['denoise']}")
    
    # Node 100: FluxGuidance - CFG scale
    if "100" in workflow:
        workflow["100"]["inputs"]["guidance"] = params['cfg']
        print(f"✓ Guidance: CFG={params['cfg']}")
    
    # Node 161: LoRA strength
    if "161" in workflow:
        workflow["161"]["inputs"]["strength_model"] = params['lora_strength']
        print(f"✓ LoRA: strength={params['lora_strength']}")
    
    # Node 135: Resolution
    if "135" in workflow:
        workflow["135"]["inputs"]["value"] = params['megapixels']
        print(f"✓ Resolution: {params['megapixels']}MP")
    
    print(f"{'='*60}\n")
    
    # Save optimized workflow
    if output_path is None:
        output_path = workflow_path.replace('.json', f'_{quality_preset}.json')
    
    with open(output_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"✓ Saved optimized workflow: {output_path}")
    
    # Print expected performance
    print(f"\nExpected performance:")
    print(f"  Quality: {quality_preset.upper()}")
    print(f"  Time per image: ~{params['steps'] * 10}s on H100")
    print(f"  Total for 49 rows: ~{params['steps'] * 10 * 49 / 60:.1f} minutes")
    print(f"  Output resolution: {params['megapixels']}MP")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Optimize Flux2 Klein face swap workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Presets:
  fast     - 4 steps, 2MP (current) - ~2 min total
  balanced - 8 steps, 4MP          - ~7 min total  
  high     - 12 steps, 6MP         - ~10 min total
  ultra    - 20 steps, 8.3MP       - ~17 min total

Examples:
  # Create high quality version
  python3 optimize_workflow.py --quality high
  
  # Create ultra quality version
  python3 optimize_workflow.py --quality ultra --output workflow_ultra.json
        """
    )
    
    parser.add_argument('--workflow', 
                       default='Flux2 Klein 9b Face Swap(API).json',
                       help='Input workflow JSON path')
    parser.add_argument('--output',
                       default=None,
                       help='Output workflow path (default: input_<quality>.json)')
    parser.add_argument('--quality',
                       choices=['fast', 'balanced', 'high', 'ultra'],
                       default='high',
                       help='Quality preset (default: high)')
    
    args = parser.parse_args()
    
    optimize_workflow(args.workflow, args.output, args.quality)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Test on GPU server:")
    print(f"   python3 test_resolutions.py --workflow <optimized_workflow> --row 1")
    print(f"\n2. If satisfied, process all rows:")
    print(f"   python3 face_swap_final.py --workflow <optimized_workflow> --csv <csv>")
    print("="*60)


if __name__ == "__main__":
    main()
