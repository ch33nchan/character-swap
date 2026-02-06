#!/usr/bin/env python3
"""
Patch Ostris AI-Toolkit for Flux: low_cpu_mem_usage=False; remove torch_dtype from tokenizer calls (fixes 'not a string').
Run: python3 scripts/patch_ai_toolkit_flux_meta.py [--work-dir lora_training]
"""
import argparse
import re
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
        return 1
    text = toolkit_file.read_text()

    flux_start = text.find("elif self.model_config.is_flux:")
    if flux_start >= 0:
        flux_section = text[flux_start : flux_start + 15000]
        if "tokenizer_2 = T5TokenizerFast.from_pretrained" in flux_section and "torch_dtype=dtype" in flux_section:
            did_replace = False
            for old, new in [
                ("tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder=\"tokenizer_2\", torch_dtype=dtype)", "tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder=\"tokenizer_2\")"),
                (" tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder=\"tokenizer_2\", torch_dtype=dtype)", " tokenizer_2 = T5TokenizerFast.from_pretrained(base_model_path, subfolder=\"tokenizer_2\")"),
            ]:
                if old in text:
                    text = text.replace(old, new, 1)
                    did_replace = True
                    break
            if did_replace:
                toolkit_file.write_text(text)
                print("Patched: removed torch_dtype from T5TokenizerFast.from_pretrained (fixes 'not a string').")
                text = toolkit_file.read_text()

    idx = text.find("FluxTransformer2DModel.from_pretrained(")
    if idx == -1:
        print("FluxTransformer2DModel.from_pretrained not found.")
        return 1
    block = text[idx : idx + 700]
    if "low_cpu_mem_usage=False" in block and "# low_cpu_mem_usage" not in block:
        print("Already patched (low_cpu_mem_usage=False in Flux block).")
        return 0

    flux_section = text[idx : idx + 1200]
    def patch_in_flux(repl):
        return text[:idx] + flux_section.replace(repl[0], repl[1], 1) + text[idx + 1200:]
    if " # low_cpu_mem_usage=False," in flux_section:
        text = patch_in_flux((" # low_cpu_mem_usage=False,", " low_cpu_mem_usage=False,"))
        toolkit_file.write_text(text)
        print("Patched: uncommented low_cpu_mem_usage=False in Flux from_pretrained.")
        return 0
    if "# low_cpu_mem_usage=False," in flux_section:
        text = patch_in_flux(("# low_cpu_mem_usage=False,", "low_cpu_mem_usage=False,"))
        toolkit_file.write_text(text)
        print("Patched: uncommented low_cpu_mem_usage=False in Flux from_pretrained.")
        return 0
    for old, new in [("    # low_cpu_mem_usage=False,", "    low_cpu_mem_usage=False,"), ("        # low_cpu_mem_usage=False,", "        low_cpu_mem_usage=False,")]:
        if old in flux_section:
            text = patch_in_flux((old, new))
            toolkit_file.write_text(text)
            print("Patched: uncommented low_cpu_mem_usage=False in Flux from_pretrained.")
            return 0

    for pattern, replacement in [
        (r"(FluxTransformer2DModel\.from_pretrained\(\s*transformer_path,\s*subfolder=subfolder,\s*torch_dtype=dtype,)\s*(\))", r"\1\n        low_cpu_mem_usage=False,\n    \2"),
        (r"(torch_dtype=dtype,)\s*\n\s*(# [^\n]*\n\s*)*\s*\)", r"\1\n        low_cpu_mem_usage=False,\n    )"),
    ]:
        new_text, n = re.subn(pattern, replacement, text, count=1)
        if n:
            toolkit_file.write_text(new_text)
            print("Patched: added low_cpu_mem_usage=False to Flux from_pretrained.")
            return 0

    line_after_dtype = None
    for line in block.split("\n"):
        if "torch_dtype=dtype" in line:
            line_after_dtype = line
            break
    if line_after_dtype is None:
        print("Could not find torch_dtype=dtype in Flux block.")
        return 1
    indent = len(line_after_dtype) - len(line_after_dtype.lstrip())
    insert_line = " " * indent + "low_cpu_mem_usage=False,\n"
    target = "torch_dtype=dtype,\n"
    pos = text.find(target, idx)
    if pos == -1:
        target = "torch_dtype=dtype,"
        pos = text.find(target, idx)
        if pos != -1:
            pos = text.find("\n", pos) + 1
    else:
        pos += len(target)
    if pos != -1:
        text = text[:pos] + insert_line + text[pos:]
        toolkit_file.write_text(text)
        print("Patched: added low_cpu_mem_usage=False to Flux from_pretrained.")
        return 0
    print("Could not find insertion point.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
