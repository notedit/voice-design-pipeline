#!/usr/bin/env python3
"""音量归一化(项目内、集级增益):把同一项目里各集之间的响度差拉平。

为什么是集级而不是条级:实测(2026-08)六个项目的响度波动大头在"集与集
之间"——voice-lover 144 集的集中位响度横跨 20.3 dB,luozhenyu 六集横跨
17.7 dB,而每集内部的 p5-p95 只有 5-10 dB。集间差异是录音/上传时的增益
不一致(session 伪影),可以放心拉平;集内波动里有说话人真实的强弱变化,
逐条拍平会重蹈 cut 阶段 gap_pass_ratio 总结过的教训——把自然变化压成
千篇一律,TTS 学不出动态。所以默认只做集级增益,集内偏差原样保留;
真要收敛集内波动,用 utt_pass_ratio 部分保留(见 LoudnormConfig)。

为什么是独立的后置阶段而不是塞进 stage1/cut:cut 的 energy_floor_db 和
stage1 的 VAD 都在读绝对 dB,前置归一化会改变它们的行为,把已经 parity
验证过的输出打乱;后置在 dataset/wavs 上还能直接处理旧项目的现有数据集,
不用重切(重切会触发那 4 条 bug 修复的 diff)。

响度定义:active-speech RMS——25ms/10ms 帧 RMS,底噪取 p5,高于底噪
15 dB 的帧算有声,响度 = 有声帧 RMS 的中位数。和 prosody.py 的口径一致,
对句首尾静音长短不敏感(整段 RMS 会被静音占比带偏)。

增益一律被峰值余量钳制(peak_ceiling_db,默认 -1 dBFS),永不削波;
被钳制的集到不了目标响度,报告里逐集写明 n_clamped 和 achieved_db,
不做静默截断。--apply 就地改写 wav;逐条实际应用的增益(钳制后)记录在
报告 applied_gains 里——旧项目的 wav 重切复现不出来(bug 修复会改输出),
这份增益记录就是唯一的撤销手段(可逆到 PCM_16 量化精度)。

已知限制:utt_pass_ratio 生效时 --apply 不幂等——第二次 apply 会把集内
偏差再乘一次 ratio(d -> d*r -> d*r^2)。集级模式(utt_pass_ratio=None)
天然幂等(重测后增益≈0)。和 punct_fix 的处理方式一样,记录在案不做
防御性加锁;真的误跑了,用报告里的 applied_gains 回退。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import ProjectConfig, load_project

_FRAME_S = 0.025
_HOP_S = 0.010
_VOICED_DB = 15.0


def active_rms_db(audio, sr):
    """active-speech RMS(dBFS)和峰值(dBFS)。audio 需为 float32 单声道。"""
    peak_db = 20 * np.log10(np.abs(audio).max() + 1e-12)
    fl, hop = int(_FRAME_S * sr), int(_HOP_S * sr)
    n = (len(audio) - fl) // hop
    if n <= 0:
        return float(20 * np.log10(np.sqrt((audio ** 2).mean()) + 1e-12)), float(peak_db)
    idx = np.arange(n)[:, None] * hop + np.arange(fl)[None, :]
    rms_db = 20 * np.log10(np.sqrt((audio[idx] ** 2).mean(axis=1)) + 1e-12)
    noise = np.percentile(rms_db, 5)
    voiced = rms_db > noise + _VOICED_DB
    loud = np.median(rms_db[voiced]) if voiced.any() else np.median(rms_db)
    return float(loud), float(peak_db)


def run(cfg: ProjectConfig, dataset_dir: Path = None, apply: bool = False):
    ln = cfg.loudnorm
    dataset_dir = dataset_dir or cfg.dataset_dir
    wavs_dir = dataset_dir / "wavs"
    files = sorted(wavs_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"{wavs_dir} 里没有 wav")

    # 测量。seg_id 形如 {prefix}{idx}_{NNNN},去掉尾部序号就是集号。
    utt = {}   # sid -> (loud_db, peak_db)
    eps = {}   # ep -> [sid,...]
    for f in files:
        sid = f.stem
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        utt[sid] = active_rms_db(audio, sr)
        eps.setdefault(sid.rsplit("_", 1)[0], []).append(sid)

    ep_loud = {ep: float(np.median([utt[s][0] for s in sids]))
               for ep, sids in eps.items()}
    target = ln.target_db if ln.target_db is not None \
        else float(np.median(list(ep_loud.values())))

    if len(eps) == 1 and ln.utt_pass_ratio is None:
        print("NOTE: 只有一集,集级归一化没有可拉平的对象(全部增益=0);"
              "该项目只有 utt_pass_ratio 这个旋钮有作用。")

    # 逐条增益:集级增益 + 可选的集内条级部分收敛,再被峰值余量钳制。
    applied_gains = {}   # sid -> 实际应用的增益(dB,钳制后)——撤销靠这个
    ep_rows = {}
    for ep, sids in sorted(eps.items()):
        gain_ep = target - ep_loud[ep]
        n_clamped = 0
        achieved = []
        for sid in sids:
            loud, peak = utt[sid]
            if ln.utt_pass_ratio is None:
                gain = gain_ep
            else:
                # 集内偏差 dev 保留 ratio 比例:最终响度 = target + dev*ratio
                dev = loud - ep_loud[ep]
                gain = target - loud + dev * ln.utt_pass_ratio
            headroom = ln.peak_ceiling_db - peak
            g = min(gain, headroom)
            if g < gain - 1e-9:
                n_clamped += 1
            applied_gains[sid] = round(g, 3)
            achieved.append(loud + g)
        ep_rows[ep] = {"n_utts": len(sids),
                       "measured_db": round(ep_loud[ep], 2),
                       "gain_db": round(gain_ep, 2),
                       "n_clamped": n_clamped,
                       "achieved_db": round(float(np.median(achieved)), 2)}
        clamp_note = f" CLAMPED {n_clamped}/{len(sids)}" if n_clamped else ""
        print(f"[{ep}] measured={ep_loud[ep]:6.1f} gain={gain_ep:+5.1f} "
              f"achieved={ep_rows[ep]['achieved_db']:6.1f}{clamp_note}", flush=True)

    report = {"dataset_dir": str(dataset_dir), "n_utts": len(utt),
              "n_episodes": len(eps), "target_db": round(target, 2),
              "target_mode": "config" if ln.target_db is not None else "auto(median)",
              "utt_pass_ratio": ln.utt_pass_ratio,
              "peak_ceiling_db": ln.peak_ceiling_db,
              "applied": apply, "episodes": ep_rows,
              "applied_gains": applied_gains}
    out_path = cfg.out_path / f"loudnorm_{dataset_dir.name}.json"
    json.dump(report, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    total_clamped = sum(r["n_clamped"] for r in ep_rows.values())
    print(f"target={target:.1f} dB ({report['target_mode']}) "
          f"episodes={len(eps)} utts={len(utt)} clamped={total_clamped} -> {out_path}")

    if apply:
        for f in files:
            g = applied_gains[f.stem]
            if abs(g) < 0.01:
                continue
            audio, sr = sf.read(f, dtype="float32")
            audio = np.clip(audio * (10 ** (g / 20)), -1.0, 1.0)
            sf.write(f, audio, sr, subtype="PCM_16")
        print(f"applied gains to {sum(1 for g in applied_gains.values() if abs(g) >= 0.01)} wavs")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("dataset_dir", nargs="?", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    cfg = load_project(args.project)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    run(cfg, dataset_dir=dataset_dir, apply=args.apply)


if __name__ == "__main__":
    main()
