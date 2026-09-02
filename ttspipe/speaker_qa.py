#!/usr/bin/env python3
"""qa-speaker:导出后逐条验证说话人纯度,补 extract 段级聚类的网。

extract 的聚类是段级的:一段里混入几秒别人的声音(主持人插话、演示播放的
音频、观众提问),整段的 ECAPA 向量仍会被主说话人主导而并进主簇,段级
和整条口径都看不出来。检测必须用窗级:

- 质心:整条向量求稳健质心(均值 -> 剔除最低 10% 重算),再用整条相似度
  top 50% 条目的全部窗向量重建窗级质心。
- 扫描:2s 窗 / 0.5s 步(1s 窗对 ECAPA 太短、噪声大),窗相似度 < low_sim
  记低窗;连续低窗数 >= flag_run 的条目标记(单个低窗多为语速/气声波动,
  连续低窗才是"有一段不是这个人")。
- 分级:low_run >= severe_run 或 win_min <= severe_min 判"重度"(基本
  确定混入);其余为"轻度"(需人工试听,演示音频/远场人声会误报)。

默认只出报告(qa_speaker_report.json);--apply 仅剔除**重度**条目
(metadata/filelist 移除 + wav 删除),轻度一律留给人工。
"""
import json
import shutil

import numpy as np

from .config import ProjectConfig
from .gpu import pick_gpu


def run(cfg: ProjectConfig, dataset_dir=None, apply: bool = False):
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(pick_gpu()))
    import librosa
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    sq = cfg.speaker_qa
    dataset_dir = dataset_dir or cfg.dataset_dir
    meta_path = dataset_dir / "metadata.csv"
    meta = [l.split("|", 1) for l in
            meta_path.read_text(encoding="utf-8").strip().split("\n")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spk = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cfg.out_path / "models" / "ecapa"),
        run_opts={"device": device})

    def emb_batch(xs):
        m = max(len(x) for x in xs)
        b = torch.zeros(len(xs), m)
        for i, x in enumerate(xs):
            b[i, :len(x)] = torch.from_numpy(x)
        with torch.no_grad():
            e = spk.encode_batch(b).squeeze(1).cpu().numpy()
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    W, H = int(sq.win_s * 16000), int(sq.hop_s * 16000)
    utt_emb, win_cache = {}, {}
    for sid, _ in meta:
        x, _ = librosa.load(dataset_dir / "wavs" / f"{sid}.wav", sr=16000, mono=True)
        utt_emb[sid] = emb_batch([x])[0]
        wins, times = [], []
        for s in range(0, max(1, len(x) - W // 2), H):
            w = x[s:s + W]
            if len(w) >= W // 2 and np.sqrt(np.mean(w ** 2)) > 0.01:
                wins.append(w)
                times.append(s / 16000)
        if wins:
            win_cache[sid] = (emb_batch(wins), times)

    # 稳健质心:整条口径两轮 -> 取 top 50% 条目的窗向量重建窗级质心
    E = np.stack(list(utt_emb.values()))
    c = E.mean(0); c /= np.linalg.norm(c)
    sims = E @ c
    c = E[sims > np.percentile(sims, 10)].mean(0); c /= np.linalg.norm(c)
    order = sorted(utt_emb, key=lambda k: float(utt_emb[k] @ c), reverse=True)
    top = [u for u in order[:len(order) // 2] if u in win_cache]
    cent = np.concatenate([win_cache[u][0] for u in top]).mean(0)
    cent /= np.linalg.norm(cent)

    utts, flagged, severe = {}, [], []
    for sid, _ in meta:
        if sid not in win_cache:
            continue
        ws, times = win_cache[sid]
        s = ws @ cent
        run_len = best = 0
        for v in s:
            run_len = run_len + 1 if v < sq.low_sim else 0
            best = max(best, run_len)
        low_t = [round(t, 2) for t, v in zip(times, s) if v < sq.low_sim]
        entry = dict(utt_sim=round(float(utt_emb[sid] @ c), 3),
                     win_min=round(float(s.min()), 3), low_run=best,
                     low_times=low_t)
        is_flag = best >= sq.flag_run
        is_severe = is_flag and (best >= sq.severe_run
                                 or s.min() <= sq.severe_min)
        entry["level"] = "severe" if is_severe else ("mild" if is_flag else "ok")
        utts[sid] = entry
        if is_severe:
            severe.append(sid)
        elif is_flag:
            flagged.append(sid)

    if apply and severe:
        keep = [(sid, t) for sid, t in meta if sid not in set(severe)]
        meta_path.write_text("\n".join(f"{s}|{t}" for s, t in keep) + "\n",
                             encoding="utf-8")
        fl = dataset_dir / "filelist.txt"
        rows = [l for l in fl.read_text(encoding="utf-8").strip().split("\n")
                if l.split("|", 1)[0].split("/")[-1].removesuffix(".wav")
                not in set(severe)]
        fl.write_text("\n".join(rows) + "\n", encoding="utf-8")
        removed_dir = cfg.work_dir / "speaker_qa_removed"
        removed_dir.mkdir(parents=True, exist_ok=True)
        for sid in severe:
            shutil.move(str(dataset_dir / "wavs" / f"{sid}.wav"),
                        str(removed_dir / f"{sid}.wav"))

    report = dict(win_s=sq.win_s, low_sim=sq.low_sim, flag_run=sq.flag_run,
                  severe_run=sq.severe_run, severe_min=sq.severe_min,
                  n_total=len(utts), n_severe=len(severe), n_mild=len(flagged),
                  severe=sorted(severe), mild=sorted(flagged), utts=utts)
    rep_path = cfg.out_path / "qa_speaker_report.json"
    json.dump(report, open(rep_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"QA-SPEAKER DONE severe={len(severe)} mild={len(flagged)} "
          f"/{len(utts)}{' (severe removed)' if apply and severe else ''} "
          f"-> {rep_path}", flush=True)
    return report
