#!/usr/bin/env python3
"""按 LLM/贪心归组结果重切音频:压缩组内超长静音,拼接文本,输出最终 TTS 数据集
(dataset/wavs, metadata.csv, filelist.txt)。对应旧的各项目 apply_merge.py。

这里固化了三处只在 kuaidao/wowan 踩坑后才修的 bug——对所有项目一视同仁,
不受 ProjectConfig 控制,因为它们是修复而不是设计取舍:

1. TERMINAL 标点集必须同时收全角+半角。旧版本只收了半角 ,;:!? 和几个
   只有全角写法的符号(。、—…),中文文本里几乎全用全角标点,导致
   join_texts() 判断"上一段是否已有标点结尾"基本总是误判,叠加插入
   多余逗号(会产生"，,"这种双标点)。
2. 重跑前清空 WAV_OUT:归组变化会让 seg_id 编号挪位,不清空会在目录里
   留下不属于当前 metadata.csv 的孤立 wav。
3. EDGE_PAD 邻组裁剪:两组之间原始间隙小于 2*edge_pad 时,各自独立地
   往外扩 edge_pad 会互相吃到对方的真实内容(常见于语速快、组间几乎
   无停顿的对话类音频)。裁到两组真实边界的中点,保证谁的 pad 都不会
   越界。

MAX_GAP 的压缩方式是唯一保留可配置的地方(ProjectConfig.cut.gap_pass_ratio):
None 时压缩是"拍平"到固定 max_gap(旧四个项目的行为);给一个 0~1 的比例时,
超出 max_gap 的部分按比例保留差异,不再让所有长停顿在最终数据里长度一样
(kuaidao/wowan 用的新方式)。这个不算 bug 修复,是两种合理设计的取舍,所以
留在配置里,不写死。
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import ProjectConfig

XFADE_S = 0.02
_FRAME_S = 0.02

# 半角+全角都要收全,见模块 docstring 第 1 点。
TERMINAL = set("。！？；：、!?;:,~—…")


def energy_trim_end(audio, t_end, sr, floor_db, search_s, pad_s):
    """从声明的段结束时间 t_end 向内(往前)搜索,回退到响度最后一次
    高于 floor_db 的位置(+ 一点缓冲),用于压缩长间隙时确定真正该保留到
    哪里,而不是照搬 VAD 声明的边界(VAD 判"有没有人声活动",不是判响度;
    句尾自然渐弱时,VAD 边界常比响度真正跌破底噪晚 0.3-0.6s)。"""
    fl = int(_FRAME_S * sr)
    n_search = int(search_s * sr)
    c = int(t_end * sr)
    lo = max(0, c - n_search)
    seg = audio[lo:c]
    if len(seg) < fl:
        return t_end
    n = len(seg) // fl
    frames = seg[:n * fl].reshape(-1, fl)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-10)
    loud = np.where(rms_db > floor_db)[0]
    if len(loud) == 0:
        return t_end - search_s
    new_end = lo / sr + (loud[-1] + 1) * _FRAME_S + pad_s
    return min(new_end, t_end)


def energy_trim_start(audio, t_start, sr, floor_db, search_s, pad_s):
    """energy_trim_end 的对称版本。"""
    fl = int(_FRAME_S * sr)
    n_search = int(search_s * sr)
    c = int(t_start * sr)
    hi = min(len(audio), c + n_search)
    seg = audio[c:hi]
    if len(seg) < fl:
        return t_start
    n = len(seg) // fl
    frames = seg[:n * fl].reshape(-1, fl)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-10)
    loud = np.where(rms_db > floor_db)[0]
    if len(loud) == 0:
        return t_start + search_s
    new_start = c / sr + loud[0] * _FRAME_S - pad_s
    return max(new_start, t_start)


def ensure_tail_silence(chunk, sr, min_sil, floor_db):
    """保证 chunk 尾部有 >= min_sil 秒的静音。"静音"按响度实测(最后一个
    高于 floor_db 的 20ms 帧之后的部分),不是 VAD 声明的边界。不足时补零——
    不能往源音频里扩:邻组中点裁剪(见 run() 里的注释)正是为了不吃到下一组
    的真实内容,间隙小于 2*edge_pad 时源音频里本来就没有足够的静音可取。
    补零前对原尾部做一个 20ms 淡出,消除"底噪 -> 纯零"的台阶感。
    返回 (chunk, 是否补了零)。"""
    fl = int(_FRAME_S * sr)
    n = len(chunk) // fl
    if n == 0:
        return chunk, False
    # 帧从尾部对齐:不足一帧的残余采样落在头部。从头部对齐的话残余落在
    # 尾部且不被测量,说话一直持续到文件末尾时会把这最后 <20ms 误当静音,
    # 补零量算少。尾对齐下 tail_s 是精确的整帧静音数,只会低估不会高估。
    frames = chunk[len(chunk) - n * fl:].reshape(-1, fl)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-10)
    loud = np.where(rms_db > floor_db)[0]
    tail_s = (n - 1 - loud[-1]) * fl / sr if len(loud) else len(chunk) / sr
    if tail_s >= min_sil:
        return chunk, False
    pad_n = int(np.ceil((min_sil - tail_s) * sr))
    fade_n = int(XFADE_S * sr)
    out = np.concatenate([chunk, np.zeros(pad_n, dtype=chunk.dtype)])
    if fade_n and len(chunk) >= fade_n:
        out[len(chunk) - fade_n:len(chunk)] *= np.linspace(
            1.0, 0.0, fade_n, dtype=np.float32)
    return out, True


def join_texts(texts):
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if out and out[-1] not in TERMINAL:
            out += ","
        out += t
    return out


def splice(a, b, xfade_n):
    """20ms 交叉淡化拼接"""
    if len(a) < xfade_n or len(b) < xfade_n:
        return np.concatenate([a, b])
    fade = np.linspace(0, 1, xfade_n, dtype=np.float32)
    head, tail = a[:-xfade_n], a[-xfade_n:] * (1 - fade) + b[:xfade_n] * fade
    return np.concatenate([head, tail, b[xfade_n:]])


def cut_group(audio, members, s0, e_last, sr, cut_cfg):
    """members: [(start,end),...] 时间连续升序。s0/e_last 是调用方算好的首尾
    边界(含 edge_pad,且已裁到不越过相邻组的地盘)。返回压缩静音后的音频。"""
    xfade_n = int(cut_cfg.xfade * sr)
    out = None
    cursor = s0
    for k, (s, e) in enumerate(members):
        nxt = members[k + 1][0] if k + 1 < len(members) else None
        seg_end = e_last if nxt is None else e
        gap = (nxt - e) if nxt is not None else 0.0
        if nxt is not None and gap > cut_cfg.max_gap:
            if cut_cfg.gap_pass_ratio is None:
                target_gap = cut_cfg.max_gap
            else:
                target_gap = cut_cfg.max_gap + (gap - cut_cfg.max_gap) * cut_cfg.gap_pass_ratio
            real_e = energy_trim_end(audio, e, sr, cut_cfg.energy_floor_db,
                                      cut_cfg.energy_search_s, cut_cfg.energy_pad_s)
            real_nxt = energy_trim_start(audio, nxt, sr, cut_cfg.energy_floor_db,
                                          cut_cfg.energy_search_s, cut_cfg.energy_pad_s)
            chunk = audio[int(cursor * sr):int((real_e + target_gap / 2) * sr)]
            cursor = real_nxt - target_gap / 2
        else:
            chunk = audio[int(cursor * sr):int(seg_end * sr)]
            cursor = seg_end
        out = chunk if out is None else splice(out, chunk, xfade_n)
    return out


def run(cfg: ProjectConfig):
    cut_cfg = cfg.cut
    sr = cfg.stage1.sr

    cfg.wavs_dir.mkdir(parents=True, exist_ok=True)
    for f in cfg.wavs_dir.glob("*.wav"):
        f.unlink()

    reports = [json.load(open(p))
               for p in sorted(cfg.reports_dir.glob("report_*.json"))]
    reports = [ep for ep in reports if ep["segments"]]

    meta_rows, filelist_rows, stats = [], [], []
    all_durs = []
    for ep in reports:
        idx = ep["episode"]
        segs = {s["i"]: s for s in ep["segments"] if s["kept"]}
        groups_path = cfg.merge_dir / f"groups_{idx}.json"
        if not groups_path.exists():
            continue
        groups = json.load(open(groups_path))
        audio, file_sr = sf.read(cfg.work_dir / f"{idx}_vocals.wav", dtype="float32")
        assert file_sr == sr

        total_len = len(audio) / sr
        n = dropped = 0
        exported = []
        for g in groups:
            members = [(segs[i]["start"], segs[i]["end"]) for i in g]
            texts = [segs[i]["text"] for i in g]
            s0 = max(0.0, members[0][0] - cut_cfg.edge_pad)
            e_last = min(total_len, members[-1][1] + cut_cfg.edge_pad)
            chunk = cut_group(audio, members, s0, e_last, sr, cut_cfg)
            dur = len(chunk) / sr
            if dur < cut_cfg.min_dur or dur > cut_cfg.max_dur:
                dropped += 1
                continue
            exported.append(dict(members=members, texts=texts, s0=s0, e_last=e_last))

        # 相邻两组原始间隙若小于 2*edge_pad,首尾 pad 会互相咬到对方的真实
        # 内容;按真实边界的中点裁开,保证谁的 pad 都不会越界。
        for k in range(len(exported) - 1):
            a, b = exported[k], exported[k + 1]
            true_gap_s, true_gap_e = a["members"][-1][1], b["members"][0][0]
            if true_gap_e > true_gap_s:
                mid = (true_gap_s + true_gap_e) / 2
                a["e_last"] = min(a["e_last"], mid)
                b["s0"] = max(b["s0"], mid)

        n_padded = 0
        for item in exported:
            chunk = cut_group(audio, item["members"], item["s0"], item["e_last"], sr, cut_cfg)
            if cut_cfg.min_tail_sil is not None:
                chunk, padded = ensure_tail_silence(
                    chunk, sr, cut_cfg.min_tail_sil, cut_cfg.energy_floor_db)
                n_padded += padded
            dur = len(chunk) / sr
            n += 1
            seg_id = f"{cfg.seg_prefix}{idx}_{n:04d}"
            sf.write(cfg.wavs_dir / f"{seg_id}.wav", chunk, sr, subtype="PCM_16")
            text = join_texts(item["texts"])
            meta_rows.append(f"{seg_id}|{text}")
            filelist_rows.append(f"wavs/{seg_id}.wav|{cfg.speaker_tag}|ZH|{text}")
            all_durs.append(dur)

        stats.append({"episode": idx, "groups": len(groups), "exported": n,
                      "dropped_len": dropped, "tail_padded": n_padded})
        print(f"[{idx}] groups={len(groups)} exported={n} dropped={dropped}"
              + (f" tail_padded={n_padded}" if cut_cfg.min_tail_sil is not None else ""),
              flush=True)

    (cfg.dataset_dir / "metadata.csv").write_text("\n".join(meta_rows) + "\n", encoding="utf-8")
    (cfg.dataset_dir / "filelist.txt").write_text("\n".join(filelist_rows) + "\n", encoding="utf-8")

    d = np.array(all_durs)
    hist = {f"{lo}-{hi}s": int(((d >= lo) & (d < hi)).sum())
            for lo, hi in [(2, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30.01)]}
    summary = {"total_utts": len(d), "total_dur_s": round(float(d.sum()), 1),
               "dur_min": round(float(d.min()), 2), "dur_med": round(float(np.median(d)), 2),
               "dur_max": round(float(d.max()), 2), "hist": hist, "episodes": stats}
    json.dump(summary, open(cfg.out_path / "merge_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary["hist"], ensure_ascii=False))
    print(f"TOTAL {len(d)} utts, {d.sum():.0f}s, med={np.median(d):.1f}s")
    return summary


if __name__ == "__main__":
    import sys
    from .config import load_project
    run(load_project(sys.argv[1]))
