#!/usr/bin/env python3
"""挑当前空闲显存最多的 GPU。用 subprocess 调 nvidia-smi,不依赖 torch——
必须在 import torch 之前调用并设好 CUDA_VISIBLE_DEVICES,否则设置不生效。"""
import subprocess

import numpy as np


def pick_gpu() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        free = [int(x) for x in r.stdout.split()]
        return int(np.argmax(free))
    except Exception:
        return 0
