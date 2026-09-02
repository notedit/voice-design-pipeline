#!/usr/bin/env python3
"""全量转写+强制对齐:对 dataset/wavs 全部音频产出字级时间戳。
输出 alignments.jsonl(每行 {id, text, items:[[ch,start,end],...]}),断点续跑。
对应旧的 align_all.py,机械搬过来,6 个项目里完全没有分叉。

后续接 punct_fix.run() 校正标点,再接 silence_qa.run() 用这份时间戳筛静音异常。
"""
import json
import os
import time

import numpy as np

from .config import ProjectConfig
from .gpu import pick_gpu
from .provenance import dataset_fingerprint, stamp


def wait_for_gpu(min_free_mb=12000, poll_s=60):
    import subprocess

    def free_mem():
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
        return [int(x) for x in r.stdout.split()]

    while True:
        free = free_mem()
        best = int(np.argmax(free))
        if free[best] >= min_free_mb:
            print(f"[gpu] using {best} ({free[best]} MiB free)", flush=True)
            return best
        print(f"[gpu] all busy (max free {max(free)} MiB), wait {poll_s}s", flush=True)
        time.sleep(poll_s)


def run(cfg: ProjectConfig, batch_size: int = 8):
    out_path = cfg.alignments_path
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(wait_for_gpu())
    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")

    import torch
    from qwen_asr import Qwen3ASRModel

    model = None
    for attempt in range(60):
        try:
            model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
                max_inference_batch_size=8, max_new_tokens=512,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"))
            break
        except Exception as e:
            print(f"model load failed ({type(e).__name__}), retry in 120s", flush=True)
            time.sleep(120)
    assert model is not None

    meta = dict(l.split("|", 1) for l in
                open(cfg.dataset_dir / "metadata.csv").read().strip().split("\n"))
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    ids = [i for i in sorted(meta) if i not in done]
    print(f"total={len(meta)} done={len(done)} remaining={len(ids)}", flush=True)

    fout = open(out_path, "a", encoding="utf-8")
    for k in range(0, len(ids), batch_size):
        chunk = ids[k:k + batch_size]
        results = None
        for attempt in range(30):
            try:
                results = model.transcribe(
                    audio=[str(cfg.wavs_dir / f"{sid}.wav") for sid in chunk],
                    language="Chinese", return_time_stamps=True)
                break
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"batch failed ({type(e).__name__}), retry in 90s", flush=True)
                time.sleep(90)
        if results is None:
            print(f"giving up batch at {k}", flush=True)
            continue
        for sid, res in zip(chunk, results):
            items = [[it.text, round(it.start_time, 3), round(it.end_time, 3)]
                     for it in (res.time_stamps or [])]
            fout.write(json.dumps({"id": sid, "text": res.text, "items": items},
                                  ensure_ascii=False) + "\n")
        fout.flush()
        if (k // batch_size) % 20 == 0:
            print(f"{min(k + batch_size, len(ids))}/{len(ids)}", flush=True)
    fout.close()
    stamp(cfg.out_path, "align", {"dataset": dataset_fingerprint(cfg.dataset_dir)})
    print("ALIGN DONE", flush=True)


if __name__ == "__main__":
    import sys
    from .config import load_project
    run(load_project(sys.argv[1]))
