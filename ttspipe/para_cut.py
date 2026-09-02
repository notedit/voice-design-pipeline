#!/usr/bin/env python3
"""para_cut:剪除对齐 gap 内未转写的声音(笑声/语气声/感叹),不动转写文本。

判定只用响度 + 对齐时间戳,不引入模型:
- 候选 = 对齐 gap(句首/句中/句尾)内,20ms 帧能量高于 cut.energy_floor_db、
  持续 >= min_cut 的连续段。文本里没有这段声音,剪掉后文本-音频反而一致。
- 刀口安全:距相邻对齐字边界 >= guard;事件两端必须各有至少一帧静音
  (声音与语音能量连续、找不到静音谷的,放弃剪除并记入 skipped)。
- 拼接:等功率 crossfade(cut.xfade);句中剪后残余停顿不足
  min_residual_pause 时补零;句尾剪后重新套用 cut.min_tail_sil 补零规则。
- 校验:剪后逐条复测,新 gap 内不得再有 >= min_cut 的可听段,复测失败
  整条回退不写。

默认只出报告 + 预览 wav(work/para_cut_preview/),--apply 才就地改写
dataset/wavs 并从 alignments.jsonl 删除受影响条目(之后需重跑 align 增量
补齐)。边笑边说(声音与对齐字重叠)不在本阶段能力内,由 llm_audit 兜底。
"""
import json
import shutil

import numpy as np
import soundfile as sf

from .config import ProjectConfig

FRAME = 0.02  # 20ms 能量帧,与 silence_qa/响度分析口径一致


def frame_db(x, sr):
    n = int(FRAME * sr)
    m = len(x) // n
    if m == 0:
        return np.array([-120.0])
    rms = np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1))
    with np.errstate(divide="ignore"):
        return np.where(rms > 0, 20 * np.log10(rms), -120.0)


def find_events(db, w0, w1, floor, min_cut, merge_gap):
    """在帧窗口 [w0,w1)(帧下标)内找能量高于 floor 的事件段,
    返回 [(f0,f1), ...](帧下标,含 f0 不含 f1),相邻段间隔 < merge_gap 合并。"""
    if w1 <= w0:
        return []
    mask = db[w0:w1] > floor
    runs, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start + w0, i + w0])
            start = None
    if start is not None:
        runs.append([start + w0, w1])
    merged = []
    mg = int(round(merge_gap / FRAME))
    for r in runs:
        if merged and r[0] - merged[-1][1] < mg:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    mc = int(round(min_cut / FRAME))
    return [(a, b) for a, b in merged if b - a >= mc]


def plan_utt(x, sr, items, cut_cfg, pc_cfg):
    """给一条音频出剪除计划。返回 (edits, skipped)。
    edits: [{kind, t0, t1, pad}](秒,已含刀口精修,pad=剪后在刀口补的静音秒数)
    skipped: [{kind, t0, t1, reason}]"""
    db = frame_db(x, sr)
    dur = len(x) / sr
    nf = len(db)
    floor = cut_cfg.energy_floor_db
    g = int(round(pc_cfg.guard / FRAME))

    def f(t):  # 秒 -> 帧下标(截断)
        return min(max(int(t / FRAME), 0), nf)

    gaps = [("head", 0.0, items[0][1])]
    for k in range(len(items) - 1):
        if items[k + 1][1] - items[k][2] > 2 * pc_cfg.guard + pc_cfg.min_cut:
            gaps.append(("mid", items[k][2], items[k + 1][1]))
    gaps.append(("tail", items[-1][2], dur))

    edits, skipped = [], []
    for kind, g0, g1 in gaps:
        w0 = f(g0) + (g if kind != "head" else 0)
        w1 = f(g1) - (g if kind != "tail" else 0)
        for a, b in find_events(db, w0, w1, floor, pc_cfg.min_cut, pc_cfg.merge_gap):
            # 事件贴到窗口内边界 = 声音与相邻语音连续,没有静音谷,不剪。
            # 句首贴文件头 / 句尾贴文件尾是合法刀口(直接截掉)。
            if a <= w0 and not (kind == "head" and w0 == 0):
                skipped.append(dict(kind=kind, t0=a * FRAME, t1=b * FRAME,
                                    reason="no_valley_left"))
                continue
            if b >= w1 and not (kind == "tail" and w1 == nf):
                skipped.append(dict(kind=kind, t0=a * FRAME, t1=b * FRAME,
                                    reason="no_valley_right"))
                continue
            t0 = max(a * FRAME - FRAME, g0 if kind != "head" else 0.0)
            t1 = min(b * FRAME + FRAME, g1 if kind != "tail" else dur)
            pad = 0.0
            if kind == "mid":
                residual = (t0 - g0) + (g1 - t1)
                pad = max(0.0, pc_cfg.min_residual_pause - residual)
            edits.append(dict(kind=kind, t0=round(t0, 3), t1=round(t1, 3),
                              pad=round(pad, 3)))
    return edits, skipped


def apply_edits(x, sr, edits, xfade_s):
    """按计划剪除(从后往前),刀口 crossfade,句中不足的停顿补零。"""
    xf = max(int(xfade_s * sr), 8)
    for e in sorted(edits, key=lambda e: -e["t0"]):
        a, b = int(e["t0"] * sr), int(e["t1"] * sr)
        a, b = max(a, 0), min(b, len(x))
        head, tail = x[:a], x[b:]
        if e["pad"] > 0:
            tail = np.concatenate([np.zeros(int(e["pad"] * sr), dtype=x.dtype), tail])
        k = min(xf, len(head), len(tail))
        if k > 0:
            fade = np.linspace(1.0, 0.0, k, dtype=x.dtype)
            joint = head[-k:] * fade + tail[:k] * (1.0 - fade)
            x = np.concatenate([head[:-k], joint, tail[k:]])
        else:
            x = np.concatenate([head, tail])
    return x


def ensure_tail_sil(x, sr, min_tail_sil, floor):
    """句尾静音保底:与 cut.py 同语义,尾部实测静音不足时补零(20ms 淡出)。"""
    if not min_tail_sil:
        return x
    db = frame_db(x, sr)
    voiced = np.nonzero(db > floor)[0]
    tail_sil = (len(db) - 1 - voiced[-1]) * FRAME if len(voiced) else len(db) * FRAME
    if tail_sil >= min_tail_sil:
        return x
    need = int((min_tail_sil - tail_sil) * sr)
    k = min(int(0.02 * sr), len(x))
    x = x.copy()
    x[-k:] *= np.linspace(1.0, 0.0, k, dtype=x.dtype)
    return np.concatenate([x, np.zeros(need, dtype=x.dtype)])


def shift_items(items, edits):
    """把对齐时间戳平移到剪后坐标,用于剪后复测。"""
    out = []
    for ch, s, e in items:
        ds = de = 0.0
        for ed in edits:
            cut = ed["t1"] - ed["t0"] - ed["pad"]
            if ed["t1"] <= s:
                ds += cut
            if ed["t1"] <= e:
                de += cut
        out.append([ch, s - ds, e - de])
    return out


def verify(x, sr, items, cut_cfg, pc_cfg):
    """剪后复测:新 gap 内不得再有可剪事件(可听未转写声音)。"""
    edits, _ = plan_utt(x, sr, items, cut_cfg, pc_cfg)
    return len(edits) == 0, edits


def run(cfg: ProjectConfig, apply: bool = False):
    pc, cc = cfg.para_cut, cfg.cut
    aligns = {}
    for line in open(cfg.alignments_path, encoding="utf-8"):
        d = json.loads(line)
        aligns[d["id"]] = d
    meta_ids = [l.split("|", 1)[0] for l in
                open(cfg.dataset_dir / "metadata.csv", encoding="utf-8").read()
                .strip().split("\n")]
    preview_dir = cfg.work_dir / "para_cut_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    report = dict(n_total=0, n_edited=0, n_skipped_only=0, n_verify_failed=0,
                  total_cut_s=0.0, utts={})
    for sid in meta_ids:
        d = aligns.get(sid)
        if d is None or len(d.get("items", [])) < 2:
            continue
        report["n_total"] += 1
        wav_path = cfg.wavs_dir / f"{sid}.wav"
        x, sr = sf.read(wav_path, dtype="float32")
        edits, skipped = plan_utt(x, sr, d["items"], cc, pc)
        if not edits:
            if skipped:
                report["n_skipped_only"] += 1
                report["utts"][sid] = dict(edits=[], skipped=skipped)
            continue
        y = apply_edits(x, sr, edits, cc.xfade)
        if any(e["kind"] == "tail" for e in edits):
            y = ensure_tail_sil(y, sr, cc.min_tail_sil, cc.energy_floor_db)
        # 头/尾被剪出新的硬边界时,重新套用 cut 阶段的边缘淡入淡出
        has_head = any(e["kind"] == "head" for e in edits)
        has_tail = any(e["kind"] == "tail" for e in edits)
        if has_head or has_tail:
            from .cut import apply_edge_fades
            y = apply_edge_fades(y, sr,
                                 cc.fade_in if has_head else 0.0,
                                 cc.fade_out if has_tail else 0.0)
        ok, residue = verify(y, sr, shift_items(d["items"], edits), cc, pc)
        cut_s = sum(e["t1"] - e["t0"] for e in edits)
        entry = dict(edits=edits, skipped=skipped, cut_s=round(cut_s, 2),
                     old_dur=round(len(x) / sr, 2), new_dur=round(len(y) / sr, 2),
                     verify_ok=ok,
                     below_min_dur=(len(y) / sr) < cc.min_dur)
        if not ok:
            entry["residue"] = residue
            report["n_verify_failed"] += 1
            report["utts"][sid] = entry
            print(f"[{sid}] VERIFY FAILED, rolled back ({len(residue)} residues)",
                  flush=True)
            continue
        sf.write(preview_dir / f"{sid}.wav", y, sr)
        report["n_edited"] += 1
        report["total_cut_s"] += cut_s
        report["utts"][sid] = entry
        kinds = ",".join(f"{e['kind']}@{e['t0']:.2f}-{e['t1']:.2f}" for e in edits)
        print(f"[{sid}] cut {cut_s:.2f}s ({kinds})"
              f"{' [BELOW MIN_DUR]' if entry['below_min_dur'] else ''}", flush=True)

        if apply:
            shutil.copy(preview_dir / f"{sid}.wav", wav_path)

    report["total_cut_s"] = round(report["total_cut_s"], 2)
    rep_path = cfg.out_path / "para_cut_report.json"
    json.dump(report, open(rep_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    if apply and report["n_edited"]:
        edited = {sid for sid, e in report["utts"].items()
                  if e.get("edits") and e.get("verify_ok")}
        kept = [line for line in open(cfg.alignments_path, encoding="utf-8")
                if json.loads(line)["id"] not in edited]
        with open(cfg.alignments_path, "w", encoding="utf-8") as fo:
            fo.writelines(kept)
        print(f"[apply] {len(edited)} wavs rewritten; their alignments removed — "
              f"run `align` to re-align them", flush=True)

    print(f"PARA-CUT DONE edited={report['n_edited']}/{report['n_total']} "
          f"cut={report['total_cut_s']}s verify_failed={report['n_verify_failed']} "
          f"-> {rep_path}", flush=True)
    return report
