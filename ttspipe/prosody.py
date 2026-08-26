#!/usr/bin/env python3
"""计算每条音频的韵律变化指标(声学特征,和 llm_audit.py 的主观打分是两回事),
写 dataset/prosody.jsonl。对应旧的 run_prosody.py + prosody_metrics.py 里
可复用的 prosody_one() 核心计算——原 prosody_metrics.py 是给另一套项目
(deepseek_aug.jsonl + 整数 id + wavs_NNN 目录批次)写的工具,和这里
metadata.csv(字符串 id)的结构对不上,只把 prosody_one() 这段真正通用的
计算逻辑搬进来,输入输出按 ProjectConfig 的 dataset 结构重写。

每条输出:
  f0_mean_hz    基频均值(音高高低)
  f0_std_st     基频标准差,半音(音高起伏强度,韵律变化核心指标)
  f0_range_st   基频 P5-P95 跨度,半音(音高动态范围)
  energy_std_db 浊音帧能量标准差 dB(重音/强弱对比)
  pause_ratio   语音段内非发声时长占比(节奏疏密)
  cps           语速 字/秒
  prosody_score 综合韵律分 = f0_std_st/energy_std_db/pause_ratio 三者
                批内 z-score 的均值(0 为批内平均,正=起伏大,负=平)
  prosody_level 1-5 等级,按 prosody_score 五分位划分,每级约 20%

全数据集当一个批次做 z-score 归一化(假定单一说话人,跨集可比)。

可选 --apply:剔除语速过快的条目(cps 上界用 IQR,Q3+1.5*IQR,和
silence_qa.py 同样的"数据驱动不拍脑袋"做法——语速快慢很大程度是文本内容
决定的,不同项目/不同说话人的自然语速中枢不一样,固定阈值(比如"cps>7")
会在语速本来就快的项目里错杀太多)。剔除方式和 silence_qa.py 一致:
metadata.csv/filelist.txt 先备份成 _preprosody 版本再重写,已删的 wav
不可恢复,靠这份备份加 report 里的 flagged 列表回退。
"""
import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf

from .config import ProjectConfig, load_project


def prosody_one(args):
    path, nchars = args
    try:
        x, sr = sf.read(path)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float64)

        # --- F0 (pyworld dio+stonemask, 5ms 帧) ---
        f0, t = pyworld.dio(x, sr, f0_floor=80.0, f0_ceil=600.0)
        f0 = pyworld.stonemask(x, f0, t, sr)
        v = f0[f0 > 0]
        if len(v) < 10:
            return path, None, "no_voiced_f0"
        st = 12 * np.log2(v / np.median(v))
        f0_std_st = float(st.std())
        f0_range_st = float(np.percentile(st, 95) - np.percentile(st, 5))

        # --- 能量 (25ms/10ms 帧 RMS) ---
        fl, hop = int(0.025 * sr), int(0.010 * sr)
        n = (len(x) - fl) // hop
        fr = np.lib.stride_tricks.as_strided(
            x, (n, fl), (x.strides[0] * hop, x.strides[0]))
        rms_db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1) + 1e-12))
        noise = np.percentile(rms_db, 5)
        voiced = rms_db > noise + 15
        idx = np.where(voiced)[0]
        energy_std_db = float(rms_db[voiced].std()) if voiced.any() else 0.0

        # --- 节奏 ---
        span = voiced[idx[0]:idx[-1] + 1]
        pause_ratio = float(1 - span.mean())
        speech_dur = len(span) * hop / sr
        cps = nchars / speech_dur if (nchars and speech_dur > 0.2) else None

        return path, dict(
            f0_mean_hz=round(float(v.mean()), 1),
            f0_std_st=round(f0_std_st, 3),
            f0_range_st=round(f0_range_st, 3),
            energy_std_db=round(energy_std_db, 3),
            pause_ratio=round(pause_ratio, 4),
            cps=round(cps, 2) if cps else None,
        ), None
    except Exception as e:
        return path, None, f"{type(e).__name__}:{e}"


def iqr_bound(arr):
    q1, q3 = np.percentile(arr, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


def flag_fast(results, sid_of_path):
    """cps 上界用 IQR(Q3+1.5*IQR),按本次分析的数据集自算,不跨项目共用。
    没有 cps(文本太短/静音过长)的条目不参与定界,也不会被标记。"""
    cps_vals = np.array([m["cps"] for m in results.values() if m["cps"] is not None])
    if len(cps_vals) < 4:
        return {}, None
    bound = iqr_bound(cps_vals)
    flagged = {}
    for p, m in results.items():
        if m["cps"] is not None and m["cps"] > bound:
            flagged[sid_of_path[p]] = [f"cps={m['cps']:.2f}>{bound:.2f}"]
    return flagged, bound


def apply_removal(dataset_dir: Path, flagged: dict, speaker_tag: str):
    for name, bak in (("metadata.csv", "metadata_preprosody.csv"),
                      ("filelist.txt", "filelist_preprosody.txt")):
        src, dst = dataset_dir / name, dataset_dir / bak
        if src.exists() and not dst.exists():
            src.rename(dst)
    old_meta = dict(l.split("|", 1) for l in
                    open(dataset_dir / "metadata_preprosody.csv").read().strip().split("\n"))
    keep_ids = [sid for sid in sorted(old_meta) if sid not in flagged]
    meta_rows = [f"{sid}|{old_meta[sid]}" for sid in keep_ids]
    fl_rows = [f"wavs/{sid}.wav|{speaker_tag}|ZH|{old_meta[sid]}" for sid in keep_ids]
    (dataset_dir / "metadata.csv").write_text("\n".join(meta_rows) + "\n", encoding="utf-8")
    (dataset_dir / "filelist.txt").write_text("\n".join(fl_rows) + "\n", encoding="utf-8")
    n_removed = 0
    for sid in flagged:
        wav = dataset_dir / "wavs" / f"{sid}.wav"
        if wav.exists():
            wav.unlink()
            n_removed += 1
    return len(keep_ids), n_removed


def run(cfg: ProjectConfig, dataset_dir=None, workers: int = 16, apply: bool = False):
    dataset_dir = dataset_dir or cfg.dataset_dir
    wavs_dir = dataset_dir / "wavs"
    meta = dict(l.split("|", 1) for l in
                open(dataset_dir / "metadata.csv").read().strip().split("\n"))

    jobs, sid_of_path = [], {}
    for sid, text in meta.items():
        p = wavs_dir / f"{sid}.wav"
        if not p.exists():
            continue
        nchars = len(re.sub(r"[^一-鿿A-Za-z0-9]", "", text))
        jobs.append((str(p), nchars))
        sid_of_path[str(p)] = sid
    print(f"共 {len(jobs)} 条音频", flush=True)

    results, errors = {}, []
    with ProcessPoolExecutor(workers) as ex:
        for i, (path, m, err) in enumerate(ex.map(prosody_one, jobs, chunksize=8)):
            if err:
                errors.append((path, err))
            else:
                results[path] = m
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    paths = list(results.keys())
    comps = []
    for key in ("f0_std_st", "energy_std_db", "pause_ratio"):
        v = np.array([results[p][key] for p in paths])
        mu, sd = v.mean(), v.std() + 1e-9
        comps.append((v - mu) / sd)
    score = np.mean(comps, axis=0)
    qs = np.percentile(score, [20, 40, 60, 80])
    for p, s in zip(paths, score):
        results[p]["prosody_score"] = round(float(s), 3)
        results[p]["prosody_level"] = int(np.searchsorted(qs, s) + 1)

    out_path = dataset_dir / "prosody.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in sorted(paths, key=lambda p: sid_of_path[p]):
            sid = sid_of_path[p]
            row = {"id": sid, "text": meta[sid], **results[p]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已写 {out_path}({len(paths)} 条)")

    if errors:
        print(f"失败 {len(errors)} 条:")
        for p, e in errors:
            print(f"  {sid_of_path.get(p, p)}: {e}")

    if paths:
        fs = np.array([results[p]["f0_std_st"] for p in paths])
        sc = np.array([results[p]["prosody_score"] for p in paths])
        cps = np.array([results[p]["cps"] for p in paths if results[p]["cps"] is not None])
        print(f"n={len(paths)}  f0_std_st 均值 {fs.mean():.2f} P5 {np.percentile(fs,5):.2f} "
              f"P95 {np.percentile(fs,95):.2f}  score P5 {np.percentile(sc,5):.2f} "
              f"P95 {np.percentile(sc,95):.2f}"
              + (f"  cps 均值 {cps.mean():.2f}" if len(cps) else ""))

    flagged, cps_bound = flag_fast(results, sid_of_path)
    fast_report = {"dataset_dir": str(dataset_dir), "n_total": len(paths),
                   "cps_bound": round(cps_bound, 3) if cps_bound is not None else None,
                   "n_flagged": len(flagged), "flagged": flagged, "applied": apply}
    fast_out = cfg.out_path / f"prosody_fast_outliers_{dataset_dir.name}.json"
    json.dump(fast_report, open(fast_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if cps_bound is not None:
        print(f"语速过快: {len(flagged)}/{len(paths)} 条超过 cps>{cps_bound:.2f} -> {fast_out}")
    if apply and flagged:
        n_kept, n_removed = apply_removal(dataset_dir, flagged, cfg.speaker_tag)
        print(f"removed {n_removed} wavs, kept {n_kept}")
    return fast_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("dataset_dir", nargs="?", default=None)
    ap.add_argument("--apply", action="store_true", help="剔除语速过快的条目")
    args = ap.parse_args()
    cfg = load_project(args.project)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    run(cfg, dataset_dir=dataset_dir, apply=args.apply)


if __name__ == "__main__":
    main()
