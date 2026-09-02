import numpy as np
import pytest
import soundfile as sf

from ttspipe.provenance import StaleArtifact, dataset_fingerprint, require_fresh, stamp


def make_dataset(root, n=3, sr=16000):
    (root / "dataset" / "wavs").mkdir(parents=True)
    rows = []
    for k in range(n):
        sid = f"t000_{k:04d}"
        sf.write(root / "dataset" / "wavs" / f"{sid}.wav", np.zeros(sr * (k + 1), np.float32), sr)
        rows.append(f"{sid}|文本{k}")
    (root / "dataset" / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root / "dataset"


def test_fresh_after_stamp_and_after_removal(tmp_path):
    ds = make_dataset(tmp_path)
    stamp(tmp_path, "align", {"dataset": dataset_fingerprint(ds)})
    require_fresh(tmp_path, "align", {"dataset": dataset_fingerprint(ds)}, "x")
    # qa 剔除一条 + fix-punct 改文本:都不动音频时间结构 -> 仍然新鲜
    lines = (ds / "metadata.csv").read_text().strip().split("\n")
    lines[0] = lines[0].split("|")[0] + "|改过的文本，"
    (ds / "metadata.csv").write_text("\n".join(lines[:-1]) + "\n")
    require_fresh(tmp_path, "align", {"dataset": dataset_fingerprint(ds)}, "x")


def test_stale_when_wav_length_changes(tmp_path):
    ds = make_dataset(tmp_path)
    stamp(tmp_path, "align", {"dataset": dataset_fingerprint(ds)})
    sf.write(ds / "wavs" / "t000_0001.wav", np.zeros(16000, np.float32), 16000)  # 剪短了
    with pytest.raises(StaleArtifact, match="时长"):
        require_fresh(tmp_path, "align", {"dataset": dataset_fingerprint(ds)}, "x")


def test_missing_stamp_is_stale(tmp_path):
    ds = make_dataset(tmp_path)
    with pytest.raises(StaleArtifact, match="stamp"):
        require_fresh(tmp_path, "align", {"dataset": dataset_fingerprint(ds)}, "x")
