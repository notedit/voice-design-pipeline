#!/usr/bin/env python3
"""denoise:基于 ClearerVoice-Studio(默认 MossFormer2_SE_48K)的语音增强,
可选后置阶段,跑在 dataset/wavs 上,默认关(逐项目按需开)。

设计取舍:
- 独立后置而不是塞进 stage1——VAD/cut/para-cut 的 energy_floor_db 读的是
  绝对响度,前置降噪会改变它们的语义;后置还能直接处理已有数据集。
- 顺序在 para-cut 之后:para-cut 的静音谷判定要在原始底噪上做。降噪是
  逐采样对齐的变换,不改时间结构,alignments.jsonl 保持有效。
- 音色保真验收:每条降噪前后各算 ECAPA 说话人向量,余弦相似度 <
  spk_sim_th 的条目判"音色受损"——--apply 时保留原音频,只在报告里列出,
  绝不静默替换。这是 TTS 数据降噪与普通降噪的根本区别:宁可留噪,不伤音色。
- 报告含每条的底噪估计(能量最低 20% 帧的均值 dB)前后对比,量化降噪量。

默认只出报告 + 预览(work/denoise_preview/),--apply 才就地改写通过
验收的条目。
"""
import json
import os
import shutil

import numpy as np
import soundfile as sf

from .config import ProjectConfig
from .gpu import pick_gpu

FRAME = 0.02


def noise_floor_db(x, sr):
    """底噪估计:能量最低 20% 的 20ms 帧的均值 dB。"""
    n = int(FRAME * sr)
    m = len(x) // n
    if m < 5:
        return -120.0
    rms = np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1))
    with np.errstate(divide="ignore"):
        db = np.where(rms > 0, 20 * np.log10(rms), -120.0)
    k = max(1, m // 5)
    return float(np.mean(np.sort(db)[:k]))


def run(cfg: ProjectConfig, dataset_dir=None, apply: bool = False):
    dn = cfg.denoise
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(pick_gpu()))
    dataset_dir = dataset_dir or cfg.dataset_dir
    wavs = sorted((dataset_dir / "wavs").glob("*.wav"))
    if not wavs:
        print(f"no wavs under {dataset_dir}/wavs", flush=True)
        return

    import librosa
    import torch
    from clearvoice import ClearVoice
    from speechbrain.inference.speaker import EncoderClassifier

    cv = ClearVoice(task="speech_enhancement", model_names=[dn.model])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spk = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cfg.out_path / "models" / "ecapa"),
        run_opts={"device": device})

    def embed(x16):
        with torch.no_grad():
            e = spk.encode_batch(torch.from_numpy(x16).float().unsqueeze(0))
        e = e.squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    preview_dir = cfg.work_dir / "denoise_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    report = dict(model=dn.model, spk_sim_th=dn.spk_sim_th, n_total=len(wavs),
                  n_ok=0, n_low_sim=0, utts={})
    for w in wavs:
        orig, sr = sf.read(w, dtype="float32")
        enh = cv(input_path=str(w), online_write=False)
        enh = np.asarray(enh, dtype=np.float32).squeeze()
        if enh.ndim > 1:
            enh = enh.mean(axis=0)
        if sr != 48000:
            enh = librosa.resample(enh, orig_sr=48000, target_sr=sr)
        # 长度对齐回原始采样数(重采样可能差几个 sample)
        if len(enh) < len(orig):
            enh = np.pad(enh, (0, len(orig) - len(enh)))
        else:
            enh = enh[:len(orig)]

        o16 = librosa.resample(orig, orig_sr=sr, target_sr=16000)
        e16 = librosa.resample(enh, orig_sr=sr, target_sr=16000)
        sim = float(embed(o16) @ embed(e16))
        nf_before, nf_after = noise_floor_db(orig, sr), noise_floor_db(enh, sr)
        ok = sim >= dn.spk_sim_th
        sf.write(preview_dir / w.name, enh, sr)
        report["utts"][w.stem] = dict(
            spk_sim=round(sim, 4), ok=ok,
            floor_before_db=round(nf_before, 1), floor_after_db=round(nf_after, 1),
            floor_gain_db=round(nf_before - nf_after, 1))
        report["n_ok" if ok else "n_low_sim"] += 1
        if apply and ok:
            shutil.copy(preview_dir / w.name, w)
        print(f"[{w.stem}] sim={sim:.3f} floor {nf_before:.0f}->{nf_after:.0f}dB"
              f"{'' if ok else '  LOW-SIM, kept original'}", flush=True)

    sims = [u["spk_sim"] for u in report["utts"].values()]
    gains = [u["floor_gain_db"] for u in report["utts"].values()]
    report["sim_median"] = round(float(np.median(sims)), 4)
    report["floor_gain_median_db"] = round(float(np.median(gains)), 1)
    rep_path = cfg.out_path / "denoise_report.json"
    json.dump(report, open(rep_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"DENOISE DONE ok={report['n_ok']}/{report['n_total']} "
          f"low_sim={report['n_low_sim']} sim_med={report['sim_median']} "
          f"floor_gain_med={report['floor_gain_median_db']}dB "
          f"{'(applied)' if apply else '(report only)'} -> {rep_path}", flush=True)
    return report
