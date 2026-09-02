"""合成音频端到端:report -> group -> group-apply -> export。不碰模型,
覆盖归组策略 + 重切 + 淡入淡出 + provenance stamp 的真实文件流。"""
import json

import numpy as np
import soundfile as sf

from ttspipe import cut, merge_apply, merge_prep
from ttspipe.config import CutConfig, ProjectConfig, Stage1Config


def make_project(tmp_path, sr=44100):
    out = tmp_path / "out"
    (out / "work" / "reports").mkdir(parents=True)
    rng = np.random.default_rng(1)
    # 8 段语音(4~6s),间隙 0.3~3s;总长 ~45s
    segs, t = [], 0.5
    for k in range(8):
        d = 4.0 + (k % 3)
        segs.append((t, t + d))
        t += d + [0.3, 0.6, 3.0, 0.2, 0.8, 0.3, 2.6, 0.4][k]
    x = np.zeros(int((t + 1) * sr), np.float32)
    for s, e in segs:
        a, b = int(s * sr), int(e * sr)
        x[a:b] = rng.standard_normal(b - a).astype(np.float32) * 0.2
    sf.write(out / "work" / "000_vocals.wav", x, sr)
    rep = dict(episode="000", file="x.wav", n_seg_vad=8, n_clusters=1, main_cluster=0,
               target_found=True, spk_clusters={}, kept=8, dropped_asr=0,
               segments=[dict(i=k, start=s, end=e, text=f"这是第{k}段测试文本", logprob=0.0,
                              kept=True, cluster=0) for k, (s, e) in enumerate(segs)])
    json.dump(rep, open(out / "work" / "reports" / "report_000.json", "w"), ensure_ascii=False)
    cfg = ProjectConfig(name="t", src_dir=str(tmp_path), out_root=str(out),
                        speaker_tag="t", seg_prefix="t", idx_mode="sequential",
                        cut=CutConfig(max_gap=1.2, gap_pass_ratio=0.3, min_dur=3.0,
                                      max_dur=30.0, min_tail_sil=0.3),
                        stage1=Stage1Config(sr=sr))
    return cfg, segs


def test_export_pipeline(tmp_path):
    cfg, segs = make_project(tmp_path)
    merge_prep.run(cfg)
    merge_apply.run(cfg)
    cut.run(cfg)
    meta = (cfg.dataset_dir / "metadata.csv").read_text(encoding="utf-8").strip().split("\n")
    wavs = sorted(cfg.wavs_dir.glob("*.wav"))
    assert len(meta) == len(wavs) >= 3
    for sid, text in (l.split("|", 1) for l in meta):
        y, sr = sf.read(cfg.wavs_dir / f"{sid}.wav")
        dur = len(y) / sr
        assert 3.0 <= dur <= 30.0
        assert abs(y[0]) < 1e-4 and abs(y[-1]) < 1e-4          # 首尾淡入淡出
        assert "，," not in text and "测试文本" in text          # TERMINAL 修复
    groups = json.load(open(cfg.merge_dir / "groups_000.json"))
    # 产品策略:每组 span <= 29s,且 3s 大停顿处必然断开(seg2|seg3, seg6|seg7)
    for g in groups:
        assert segs[g[-1]][1] - segs[g[0]][0] <= 29.0
    flat_breaks = {g[-1] for g in groups[:-1]}
    assert {2, 6} <= flat_breaks
    assert (cfg.out_path / "provenance" / "export.json").exists()


def test_manifest_matches_groups_and_tail_punct_uses_it(tmp_path):
    from ttspipe import tail_punct
    cfg, _ = make_project(tmp_path)
    merge_prep.run(cfg); merge_apply.run(cfg); cut.run(cfg)
    manifest = json.load(open(cfg.dataset_dir / "manifest.json"))
    groups = json.load(open(cfg.merge_dir / "groups_000.json"))
    for sid, m in manifest.items():
        assert groups[m["gid"]] == m["segs"] and m["src_end"] > m["src_start"]
    rep = tail_punct.run(cfg)
    assert rep["n_utts"] == len(manifest)
