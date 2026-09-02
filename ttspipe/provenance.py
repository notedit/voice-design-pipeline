#!/usr/bin/env python3
"""产物新鲜度(provenance):把"merge/ 与 dataset/ 是不是同代产物"、"alignments
是不是对着当前音频算的"从靠人眼判断变成机器判断。

每个关键 stage 写完产物后记一枚 stamp(out_root/provenance/<stage>.json):
自己吃的输入的指纹 + 生效配置段的 hash。下游 stage 开跑前 require_fresh():
重新算上游输入的当前指纹,和 stamp 里记的比,不一致就拒跑并说明原因。

指纹口径刻意选"时间结构"而不是字节:dataset 指纹按每条 (id, 采样数) 比。
这样 denoise/loudnorm 这类逐采样对齐的改写不会误报过期(alignments 仍
有效);fix-punct/fix-tail-punct 只改文本、不改音频,也不会误报——
alignments 是对音频算的,与 metadata 文本无关。而 remove-nonspeech 改了
时长、export 重切了音频,都会如实判为过期。metadata 里已不存在的 id
(qa 剔除)不算过期——对齐里多几条无害。metadata 文本 hash 只记录不比对。
"""
import hashlib
import json
import time
from pathlib import Path


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return _h(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else "missing"


def files_hash(paths) -> str:
    return _h("|".join(f"{Path(p).name}:{file_hash(p)}" for p in sorted(paths)))


def dataset_fingerprint(dataset_dir: Path) -> dict:
    """{id: frames} + metadata 文本 hash。"""
    import soundfile as sf
    meta_path = Path(dataset_dir) / "metadata.csv"
    if not meta_path.exists():
        return {"ids": {}, "meta": "missing"}
    ids = {}
    for line in meta_path.read_text(encoding="utf-8").strip().split("\n"):
        sid = line.split("|", 1)[0]
        wav = Path(dataset_dir) / "wavs" / f"{sid}.wav"
        ids[sid] = sf.info(str(wav)).frames if wav.exists() else -1
    return {"ids": ids, "meta": file_hash(meta_path)}


def _stamp_path(out_root: Path, stage: str) -> Path:
    return Path(out_root) / "provenance" / f"{stage}.json"


def stamp(out_root: Path, stage: str, inputs: dict, config_obj=None):
    """记录本 stage 吃的输入指纹。inputs: {name: fingerprint(任意可 json 化)}"""
    p = _stamp_path(out_root, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg_hash = _h(json.dumps(config_obj.__dict__, sort_keys=True, default=str)) \
        if config_obj is not None else None
    json.dump(dict(stage=stage, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                   config_hash=cfg_hash, inputs=inputs),
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def load_stamp(out_root: Path, stage: str):
    p = _stamp_path(out_root, stage)
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


class StaleArtifact(RuntimeError):
    pass


def require_fresh(out_root: Path, stage: str, inputs: dict, consumer: str):
    """校验 stage 的产物对当前输入仍然有效。inputs 与 stamp 时同名同口径。
    dataset 指纹按 (id, frames) 比,当前 metadata 里不存在的 id 忽略。"""
    st = load_stamp(out_root, stage)
    if st is None:
        raise StaleArtifact(
            f"[{consumer}] 缺少 {stage} 的 provenance stamp——{stage} 是用旧版代码"
            f"跑的或从未跑过;重跑 `{stage}` 后再来。")
    for name, cur in inputs.items():
        rec = st["inputs"].get(name)
        if isinstance(cur, dict) and "ids" in cur:
            if rec is None:
                raise StaleArtifact(f"[{consumer}] {stage} stamp 里没有 {name}")
            changed = [sid for sid, fr in cur["ids"].items()
                       if rec["ids"].get(sid) != fr]
            if changed:
                raise StaleArtifact(
                    f"[{consumer}] {stage} 的产物已过期:{len(changed)} 条音频"
                    f"时长/存在性变了(如 {changed[:3]})。上游改过 dataset 后"
                    f"必须重跑 `{stage}`。")
        elif rec != cur:
            raise StaleArtifact(
                f"[{consumer}] {stage} 的产物已过期:输入 {name} 与 stamp 不一致"
                f"({rec} != {cur})。重跑 `{stage}`。")
