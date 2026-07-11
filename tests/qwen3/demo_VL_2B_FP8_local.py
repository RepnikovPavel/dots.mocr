#!/usr/bin/env python3
"""Compatibility entry point for the fixed strict-offline CUDA runner.

For the CPU reference path use:
    python3 /app/chat_gpt_56_workspace/qwen3/run_cpu_offline.py
"""

import os
import sys
from pathlib import Path


# These must be set before importing Transformers or huggingface_hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

WORKSPACE = Path(__file__).resolve().parents[2] / "chat_gpt_56_workspace" / "qwen3"
if not (WORKSPACE / "qwen3_vl_offline.py").is_file():
    raise FileNotFoundError(f"fixed Qwen3-VL runner not found: {WORKSPACE}")
sys.path.insert(0, str(WORKSPACE))

from qwen3_vl_offline import main


if __name__ == "__main__":
    raise SystemExit(main("cuda"))
