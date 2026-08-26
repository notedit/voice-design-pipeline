#!/usr/bin/env python3
"""用 OpenRouter 上的 inkling-small 对 dataset/ 逐条打分+判定是否适合 TTS 训练。
每条给出韵律分/自然度分,并按 4 项标准判定 include/exclude:多说话人、异常爆音、
异常停顿、口齿不清。断点续跑(结果存 qa_llm_audit.partial.json)。

对应 kuaidao/wowan 用的 llm_audit.py schema——这是 6 个项目里最新的版本,
consolidate 掉了 voice-lover 那边的四代旧版(llm_audit.py/_all/_strict/_v4,
每代 prompt 和判定标准都不一样)。人物描述从 ProjectConfig.llm_audit_speaker_desc
读,不同项目内容不一样(kuaidao 是单人男声讲课,wowan 是两人对话,voice-lover
是单人男声,luozhenyu/dialogue-samples/wav-raw 是单人女声有声书朗读),硬编码在
prompt 里会答非所问。

用法: python -m ttspipe.llm_audit <project> [N|ids.txt]
需要 OPENROUTER_API_KEY。
"""
import base64
import io
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import requests
import soundfile as sf

from .config import ProjectConfig

MODEL = "thinkingmachines/inkling-small"
URL = "https://openrouter.ai/api/v1/chat/completions"
WORKERS = 24

PROMPT = """你是 TTS 训练数据质检员。听这段中文音频({speaker_desc}),完成两项任务。

1. 打分(1-5 整数,5 最好):
   - prosody_score:韵律自然度——语调起伏、重音、停顿节奏是否符合正常说话习惯,是否拖沓、忽快忽慢或过于平淡
   - naturalness_score:整体听感自然度——流畅度、咬字清晰度、有没有机械感或违和感

2. 判定是否适合用作 TTS 训练语料(verdict: include/exclude),命中以下任意一条即 exclude:
   - multi_speaker:混入了第二个说话人(插话、对话、画外音)——如果音频本身就是多人对话,这条不适用,忽略
   - clipping:明显爆音/破音/削波失真(声音突然爆裂、数字失真、麦克风过载)
   - abnormal_pause:异常停顿(长时间死寂或明显不自然的卡顿;正常句间停顿、首尾静音不算)
   - unclear_speech:口齿不清、含糊、吞字严重,内容难以听清

转录文本:{text}
参考:时长 {dur:.1f}s

只输出一个 JSON 对象:
{{"prosody_score": 1-5, "naturalness_score": 1-5,
"multi_speaker": true/false, "clipping": true/false,
"abnormal_pause": true/false, "unclear_speech": true/false,
"verdict": "include|exclude", "reason": "一句话中文原因(exclude 时说明命中了哪条)"}}"""

_lock = threading.Lock()


def features(path):
    a, sr = librosa.load(path, sr=16000, mono=True)
    dur = len(a) / sr
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return dur, base64.b64encode(buf.getvalue()).decode()


def audit_one(cfg: ProjectConfig, key, sid, text):
    dur, b64 = features(cfg.wavs_dir / f"{sid}.wav")
    prompt = PROMPT.format(speaker_desc=cfg.llm_audit_speaker_desc, text=text, dur=dur)
    body = {
        "model": MODEL,
        "temperature": 0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ],
        }],
    }
    for attempt in range(6):
        try:
            r = requests.post(URL, json=body, timeout=180,
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.strip("`").lstrip("json").strip()
            v = json.loads(content)
            assert v.get("verdict") in ("include", "exclude")
            return {"id": sid, "dur": round(dur, 1), **v}
        except Exception as e:
            if attempt == 5:
                return {"id": sid, "dur": round(dur, 1), "verdict": "error",
                        "reason": f"{type(e).__name__}: {e}"[:120]}
            time.sleep(8 * (attempt + 1))


def run(cfg: ProjectConfig, selector=None):
    """selector: None(全部) | int(前 N 条) | 文件路径(逐行 id 列表)"""
    key = os.environ["OPENROUTER_API_KEY"]
    meta = dict(l.split("|", 1) for l in
                open(cfg.dataset_dir / "metadata.csv").read().strip().split("\n"))
    if isinstance(selector, int):
        ids = sorted(meta)[:selector]
    elif isinstance(selector, str):
        ids = [l.strip() for l in open(selector) if l.strip()]
    else:
        ids = sorted(meta)

    part = cfg.out_path / "qa_llm_audit.partial.json"
    results = json.load(open(part)) if part.exists() else []
    done = {r["id"] for r in results if r.get("verdict") != "error"}
    results = [r for r in results if r["id"] in done]
    todo = [i for i in ids if i not in done]
    print(f"todo={len(todo)} done={len(done)}", flush=True)

    n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(audit_one, cfg, key, sid, meta[sid]): sid for sid in todo}
        for fut in as_completed(futs):
            with _lock:
                results.append(fut.result())
                n += 1
                if n % 50 == 0:
                    json.dump(results, open(part, "w", encoding="utf-8"), ensure_ascii=False)
                    print(f"{n}/{len(todo)}", flush=True)

    results.sort(key=lambda r: r["id"])
    json.dump(results, open(part, "w", encoding="utf-8"), ensure_ascii=False)
    keep_ids = set(ids) | done
    final = [r for r in results if r["id"] in keep_ids]
    json.dump(final, open(cfg.out_path / "qa_llm_audit.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("verdicts:", dict(Counter(r["verdict"] for r in final)))
    for tag in ("multi_speaker", "clipping", "abnormal_pause", "unclear_speech"):
        print(f"  {tag}: {sum(1 for r in final if r.get(tag))}")
    print("AUDIT DONE", len(final), flush=True)
    return final


if __name__ == "__main__":
    from .config import load_project
    sel = None
    if len(sys.argv) > 2:
        sel = int(sys.argv[2]) if sys.argv[2].isdigit() else sys.argv[2]
    run(load_project(sys.argv[1]), selector=sel)
