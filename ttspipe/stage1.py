#!/usr/bin/env python3
"""第一阶段:BS-RoFormer 去 BGM -> silero VAD 分段 -> ECAPA 聚类锁定目标说话人 ->
Qwen3-ASR 批量转写。每集写 work/reports/report_XXX.json(断点续跑:已有则跳过)。

目标说话人的两种选法(step_speaker / select_target_cluster):
- 默认(ref_audio=None):锁定总时长最大的簇——旧行为,和历史项目 parity。
- 参考模式(ref_audio=样本路径):对参考样本算 ECAPA 向量,挑与它余弦距离
  最近的簇;最近距离仍超过 ref_max_dist 时判定本集没有目标说话人,整集
  不保留(report 里 target_found=false)。目标不是说话最多的人时必须用它。
一集聚出多个簇时会往 work/spk_preview/<idx>/cluster_<k>.wav 导出各簇试听
预览(每簇拼 ~12s),用户听完指认后可直接把选中的预览 wav 设为 ref_audio。

模型只加载一次,四步(分离/VAD/聚类/ASR)在同一个进程里跑完一整批集数,
不拆成四个可独立调用的 CLI 阶段——拆开意味着每一步都要重新加载一遍这些
GB 级模型,不划算,现实中也从来是当一个整体跑的。
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .gpu import pick_gpu


def discover_files(cfg: ProjectConfig):
    """按 cfg.idx_mode 返回 [(idx, path), ...],按处理顺序排好。"""
    files = []
    for pattern in cfg.file_glob:
        files.extend(cfg.src_path.glob(pattern))
    if cfg.idx_mode == "prefix3":
        files = sorted(files)
        return [(p.name[:3], p) for p in files]
    elif cfg.idx_mode == "sequential":
        if cfg.episode_order:
            name_to_idx = {name: f"{i:03d}" for i, name in enumerate(cfg.episode_order)}
            files = sorted(files, key=lambda p: name_to_idx[p.stem])
            return [(name_to_idx[p.stem], p) for p in files]
        files = sorted(files)
        return [(f"{i:03d}", p) for i, p in enumerate(files)]
    raise ValueError(f"未知 idx_mode: {cfg.idx_mode}")


def has_chinese(t):
    return any("一" <= c <= "鿿" for c in t)


def select_target_cluster(embs, labels, segs, ref_emb=None, ref_max_dist=0.60):
    """从聚类结果里选目标说话人簇。纯函数,不碰模型,可单独测试。

    embs: (N, D) 已 L2 归一化的段向量;labels: 长度 N 的簇标签 list;
    ref_emb: 已归一化的参考向量,None = 按总时长选。
    返回 (目标簇标签或 None, 各簇统计 {lab: {dur, n[, ref_dist]}})。
    """
    dur, cent = {}, {}
    for lab in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == lab]
        dur[lab] = sum(segs[i][1] - segs[i][0] for i in idxs)
        c = embs[idxs].mean(axis=0)
        cent[lab] = c / np.linalg.norm(c)
    stats = {lab: {"dur": round(dur[lab], 1), "n": labels.count(lab)} for lab in dur}
    if ref_emb is None:
        return max(dur, key=dur.get), stats
    dists = {lab: float(1.0 - cent[lab] @ ref_emb) for lab in cent}
    for lab in stats:
        stats[lab]["ref_dist"] = round(dists[lab], 3)
    main = min(dists, key=dists.get)
    if dists[main] > ref_max_dist:
        return None, stats
    return main, stats


class Stage1Runner:
    """加载一次模型,跑完整批集数。"""

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.s1 = cfg.stage1
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(pick_gpu()))
        os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")

        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[init] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
              f"device={self.device}", flush=True)

        self.work_dir = cfg.work_dir
        self.reports_dir = cfg.reports_dir
        self.sep_dir = self.work_dir / "bsroformer"
        for d in (self.work_dir, self.reports_dir, self.sep_dir):
            d.mkdir(parents=True, exist_ok=True)

        from silero_vad import load_silero_vad, get_speech_timestamps
        self._get_speech_timestamps = get_speech_timestamps
        self.vad_model = load_silero_vad()

        from speechbrain.inference.speaker import EncoderClassifier
        self.spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(cfg.out_path / "models" / "ecapa"),
            run_opts={"device": self.device})

        from qwen_asr import Qwen3ASRModel
        self.asr_model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16,
            device_map="cuda:0" if self.device == "cuda" else "cpu",
            max_inference_batch_size=self.s1.asr_batch,
            max_new_tokens=512,
        )
        self._separator = None

        self.ref_emb = None
        if self.s1.ref_audio:
            self.ref_emb = self._embed_ref(self.s1.ref_audio)
            print(f"[init] target speaker ref: {self.s1.ref_audio} "
                  f"(ref_max_dist={self.s1.ref_max_dist})", flush=True)

    def _embed_ref(self, path):
        import librosa
        audio, _ = librosa.load(path, sr=self.s1.vad_sr, mono=True)
        if len(audio) < self.s1.vad_sr:
            raise ValueError(f"参考音频过短(<1s): {path}")
        with self.torch.no_grad():
            emb = self.spk_model.encode_batch(self.torch.from_numpy(audio).unsqueeze(0))
        emb = emb.squeeze().cpu().numpy()
        return emb / np.linalg.norm(emb)

    def _sh(self, cmd):
        subprocess.run(cmd, check=True, capture_output=True)

    def step_convert(self, src: Path, idx: str) -> Path:
        out = self.work_dir / f"{idx}_full.wav"
        if not out.exists():
            self._sh(["ffmpeg", "-y", "-i", str(src), "-ar", str(self.s1.sr), "-ac", "2", str(out)])
        return out

    def step_separate(self, full: Path, idx: str) -> Path:
        import soundfile as sf
        out = self.work_dir / f"{idx}_vocals.wav"
        if out.exists():
            return out
        if self._separator is None:
            from audio_separator.separator import Separator
            self._separator = Separator(output_dir=str(self.sep_dir), output_format="WAV",
                                        log_level=40)
            self._separator.load_model(self.s1.bsr_model)
        files = self._separator.separate(str(full))
        voc = next(f for f in files if "(Vocals)" in f)
        audio, sr = sf.read(self.sep_dir / voc)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        # 硬校验:源有声但人声轨接近数字静音 = 分离环节坏了(依赖被动过、
        # 模型输出错乱等),必须响亮报错,绝不能让下游安静地跑出空数据集。
        def _rms_db(x):
            r = float(np.sqrt(np.mean(x ** 2)))
            return 20 * np.log10(r) if r > 0 else -120.0
        full_audio, _ = sf.read(full)
        full_mono = full_audio.mean(axis=1) if full_audio.ndim == 2 else full_audio
        if _rms_db(full_mono) > -40.0 and _rms_db(mono) < -60.0:
            raise RuntimeError(
                f"分离输出的人声轨近乎静音(full={_rms_db(full_mono):.1f}dB, "
                f"vocals={_rms_db(mono):.1f}dB)——BS-RoFormer 输出异常,"
                f"通常是依赖环境被改动(检查 numpy/torchaudio 版本)。")
        sf.write(out, mono.astype(np.float32), sr)
        for f in files:  # 立体声中间文件较大,用完即删
            (self.sep_dir / f).unlink(missing_ok=True)
        return out

    def step_vad(self, vocals: Path):
        import librosa
        s1 = self.s1
        audio16, _ = librosa.load(vocals, sr=s1.vad_sr, mono=True)
        ts = self._get_speech_timestamps(
            self.torch.from_numpy(audio16), self.vad_model, sampling_rate=s1.vad_sr,
            min_speech_duration_ms=250, min_silence_duration_ms=150,
            speech_pad_ms=60, return_seconds=True)
        segs = []
        for t in ts:
            s, e = t["start"], t["end"]
            if segs and s - segs[-1][1] <= s1.split_gap:
                segs[-1][1] = e
            else:
                segs.append([s, e])

        def split_long(seg):
            s, e = seg
            if e - s <= s1.max_seg:
                return [seg]
            inner = [t for t in ts if t["start"] >= s and t["end"] <= e]
            gaps = [(inner[i + 1]["start"] - inner[i]["end"],
                     inner[i]["end"], inner[i + 1]["start"]) for i in range(len(inner) - 1)]
            if not gaps:
                return [seg]
            # 分级候选(split_tiers):优先在长停顿(句末/分句)下刀,逐级
            # 放宽,保底才用逗号级短停顿;None = 旧行为,所有间隙同台竞争。
            cand = gaps
            if s1.split_tiers:
                for th in s1.split_tiers:
                    tier = [g for g in gaps if g[0] >= th]
                    if tier:
                        cand = tier
                        break
            mid = s + (e - s) / 2
            _, ge, gs = min(cand, key=lambda g: abs((g[1] + g[2]) / 2 - mid) - g[0] * 2)
            return split_long([s, ge]) + split_long([gs, e])

        final = []
        for seg in segs:
            for sub in split_long(seg):
                if sub[1] - sub[0] >= s1.min_seg:
                    final.append(sub)
        return audio16, final

    def step_speaker(self, audio16, segs):
        from sklearn.cluster import AgglomerativeClustering
        s1 = self.s1
        embs = []
        for s, e in segs:
            chunk = audio16[int(s * s1.vad_sr):int(e * s1.vad_sr)]
            with self.torch.no_grad():
                emb = self.spk_model.encode_batch(self.torch.from_numpy(chunk).unsqueeze(0))
            embs.append(emb.squeeze().cpu().numpy())
        embs = np.stack(embs)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        if len(segs) == 1:
            labels = [0]
        else:
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=s1.spk_dist,
                metric="cosine", linkage="average").fit_predict(embs).tolist()
        main, stats = select_target_cluster(
            embs, labels, segs, ref_emb=self.ref_emb, ref_max_dist=s1.ref_max_dist)
        keep = set() if main is None else {i for i, lab in enumerate(labels) if lab == main}
        return keep, labels, main, stats

    def write_previews(self, audio16, segs, labels, idx, per_seg_s=5.0, total_s=12.0):
        """每个簇拼一条 ~total_s 秒的试听 wav,给人工指认目标说话人用。"""
        import soundfile as sf
        sr = self.s1.vad_sr
        pv_dir = self.work_dir / "spk_preview" / idx
        pv_dir.mkdir(parents=True, exist_ok=True)
        for lab in sorted(set(labels)):
            chunks, total = [], 0.0
            for i in [i for i, l in enumerate(labels) if l == lab]:
                s, e = segs[i]
                e = min(e, s + per_seg_s)
                chunks.append(audio16[int(s * sr):int(e * sr)])
                total += e - s
                if total >= total_s:
                    break
            sf.write(pv_dir / f"cluster_{lab}.wav",
                     np.concatenate(chunks).astype(np.float32), sr)
        return pv_dir

    def step_asr_batch(self, audio16, spans):
        s1 = self.s1
        audios = [(audio16[int(s * s1.vad_sr):int(e * s1.vad_sr)].astype(np.float32), s1.vad_sr)
                  for s, e in spans]
        texts = []
        for i in range(0, len(audios), s1.asr_batch):
            results = self.asr_model.transcribe(audio=audios[i:i + s1.asr_batch],
                                                language="Chinese")
            texts.extend(r.text.strip() for r in results)
        return texts

    def process(self, src: Path, idx: str):
        rep_path = self.reports_dir / f"report_{idx}.json"
        if rep_path.exists():
            print(f"[{idx}] skip (report exists)", flush=True)
            return
        full = self.step_convert(src, idx)
        vocals = self.step_separate(full, idx)
        audio16, segs = self.step_vad(vocals)
        if not segs:
            json.dump({"episode": idx, "file": src.name, "n_seg_vad": 0,
                       "n_clusters": 0, "main_cluster": None, "target_found": False,
                       "spk_clusters": {}, "kept": 0, "dropped_asr": 0, "segments": []},
                      open(rep_path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"[{idx}] no speech found", flush=True)
            return
        keep, labels, main, spk_stats = self.step_speaker(audio16, segs)
        n_clusters = len(spk_stats)
        if n_clusters > 1:
            pv_dir = self.write_previews(audio16, segs, labels, idx)
            print(f"[{idx}] cluster previews -> {pv_dir}", flush=True)
        if main is None:
            best = min(v["ref_dist"] for v in spk_stats.values())
            print(f"[{idx}] TARGET SPEAKER NOT FOUND: nearest cluster dist={best:.3f} "
                  f"> ref_max_dist={self.s1.ref_max_dist}, keeping nothing", flush=True)
        kept_idx = sorted(keep)
        texts = self.step_asr_batch(audio16, [segs[i] for i in kept_idx]) if kept_idx else []
        ep = {"episode": idx, "file": src.name, "n_seg_vad": len(segs),
              "n_clusters": n_clusters, "main_cluster": None if main is None else int(main),
              "target_found": main is not None,
              "spk_clusters": {str(k): v for k, v in spk_stats.items()},
              "kept": 0, "dropped_asr": 0, "segments": []}
        for i, text in zip(kept_idx, texts):
            s, e = segs[i]
            drop = (len(text) < 2) or (not has_chinese(text))
            ep["segments"].append({"i": i, "start": round(s, 2), "end": round(e, 2),
                                   "text": text, "logprob": 0.0,
                                   "kept": not drop, "cluster": labels[i]})
            if drop:
                ep["dropped_asr"] += 1
            else:
                ep["kept"] += 1
        json.dump(ep, open(rep_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        kept_dur = sum(seg["end"] - seg["start"] for seg in ep["segments"] if seg["kept"])
        print(f"[{idx}] vad={len(segs)} clusters={n_clusters} "
              f"main={len(kept_idx)} kept={ep['kept']} ({kept_dur:.0f}s) "
              f"dropped_asr={ep['dropped_asr']}", flush=True)


def run(cfg: ProjectConfig, only=None):
    only = set(only) if only else set()
    runner = Stage1Runner(cfg)
    for idx, src in discover_files(cfg):
        if only and idx not in only:
            continue
        try:
            runner.process(src, idx)
        except Exception as e:
            print(f"[{idx}] ERROR: {e}", flush=True)
    print("STAGE1 DONE", flush=True)


if __name__ == "__main__":
    import sys
    from .config import load_project
    run(load_project(sys.argv[1]), only=sys.argv[2:])
