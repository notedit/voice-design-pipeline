#!/usr/bin/env python3
"""doctor:环境自检——把 AGENTS.md"第 0 步"从文字约定变成命令。

1. 主 venv 依赖可导入 + 关键版本(numpy/torch/librosa)+ CUDA 可用;
2. 侧车 .venv-enhance 存在且 clearvoice 可导入(denoise 阶段用);
3. [给了项目时] BS-RoFormer 分离冒烟:从项目第一个源文件中段截 60s 分离,
   人声轨必须有能量——2026-08 那次 clearvoice 装进主 venv 后分离静默输出
   全零人声,就是这个检查能拦住的事故。
任一项失败退出码非 0。
"""
import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MAIN_MODS = ["torch", "silero_vad", "speechbrain", "qwen_asr", "audio_separator",
             "librosa", "soundfile", "pyworld", "sklearn", "onnxruntime", "audioread"]


def _rms_db(x):
    r = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    return 20 * np.log10(r) if r > 0 else -120.0


def check_imports():
    ok = True
    for m in MAIN_MODS:
        try:
            importlib.import_module(m)
            print(f"  [ok] {m}")
        except Exception as e:
            print(f"  [FAIL] {m}: {e}")
            ok = False
    import numpy, torch, librosa
    print(f"  numpy={numpy.__version__} torch={torch.__version__} "
          f"librosa={librosa.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  [WARN] CUDA 不可用,extract/align 会极慢")
    return ok


def check_enhance_venv(repo_root: Path):
    py = repo_root / ".venv-enhance" / "bin" / "python"
    if not py.exists():
        print("  [WARN] .venv-enhance 不存在(denoise 阶段不可用);"
              " bash scripts/setup_env.sh enhance")
        return True
    r = subprocess.run([str(py), "-c", "import clearvoice, speechbrain"],
                       capture_output=True, text=True)
    print(f"  [{'ok' if r.returncode == 0 else 'FAIL'}] .venv-enhance clearvoice")
    return r.returncode == 0


def check_separation(cfg):
    """60s 分离冒烟:人声轨要有能量。"""
    import soundfile as sf
    from .stage1 import discover_files
    files = discover_files(cfg)
    if not files:
        print("  [WARN] 项目源目录里没有音频,跳过分离冒烟")
        return True
    src = files[0][1]
    with tempfile.TemporaryDirectory() as td:
        info = sf.info(str(src)) if src.suffix.lower() == ".wav" else None
        start = max(0, (info.duration / 2 - 30)) if info else 60
        clip = Path(td) / "clip.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
                        "-t", "60", "-i", str(src), "-ar", str(cfg.stage1.sr),
                        "-ac", "2", str(clip)], check=True)
        from audio_separator.separator import Separator
        sep = Separator(output_dir=td, output_format="WAV", log_level=40)
        sep.load_model(cfg.stage1.bsr_model)
        outs = sep.separate(str(clip))
        full, _ = sf.read(clip)
        voc = next(f for f in outs if "(Vocals)" in f)
        v, _ = sf.read(Path(td) / voc)
        full_db, voc_db = _rms_db(full.mean(1)), _rms_db(v.mean(1) if v.ndim == 2 else v)
    ok = voc_db > -45 and voc_db >= full_db - 15
    print(f"  [{'ok' if ok else 'FAIL'}] 分离冒烟 {src.name}: full={full_db:.1f}dB "
          f"vocals={voc_db:.1f}dB" + ("" if ok else
          "  <- 人声轨异常安静,分离环节坏了(检查依赖版本 / 重建 .venv)"))
    return ok


def run(cfg=None):
    repo_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    print("== 1/3 主 venv 依赖 ==")
    ok = check_imports()
    print("== 2/3 增强侧车 ==")
    ok &= check_enhance_venv(repo_root)
    print("== 3/3 分离冒烟 ==")
    if cfg is None:
        print("  (未指定项目,跳过;`run.py doctor <project>` 可做 60s 分离冒烟)")
    else:
        from .gpu import pick_gpu
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(pick_gpu()))
        ok &= check_separation(cfg)
    print("DOCTOR", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)
