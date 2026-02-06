#!/usr/bin/env python3
"""
Patch Ostris AI-Toolkit to avoid meta tensor when loading Flux from HuggingFace.
Run: python3 scripts/patch_ai_toolkit_flux_meta.py [--work-dir lora_training]
"""
import argparse
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="lora_training")
    args = p.parse_args()
    work_dir = Path(__file__).resolve().parent.parent / args.work_dir
    toolkit_file = work_dir / "ai-toolkit" / "toolkit" / "stable_diffusion_model.py"

    if not toolkit_file.exists():
        print(f"Not found: {toolkit_file}")
        print("Run after: python3 train_character_lora.py setup --work-dir", args.work_dir)
        return 1
    text = toolkit_file.read_text()
    if "low_cpu_mem_usage=False" in text and "FluxTransformer2DModel" in text:
        print("Already patched.")
        return 0
    if "FluxTransformer2DModel.from_pretrained" not in text:
        print("Toolkit file structure unexpected.")
        return 1
    text = text.replace("# low_cpu_mem_usage=False,", "low_cpu_mem_usage=False,")
    if "low_cpu_mem_usage=False" not in text:
        idx = text.find("torch_dtype=dtype,")
        if idx == -1:
            print("Could not find insertion point.")
            return 1
        end = text.find("\n", idx) + 1
        text = text[:end] + "        low_cpu_mem_usage=False,\n" + text[end:]
    toolkit_file.write_text(text)
    print("Patched: Flux from_pretrained now uses low_cpu_mem_usage=False")
    return 0

if __name__ == "__main__":
    sys.exit(main())
