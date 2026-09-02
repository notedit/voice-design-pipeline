#!/usr/bin/env python3
"""项目配置模型:每个数据源(voice-lover/kuaidao/wowan/...)对应 projects/<name>.json
一份。分两类字段——

- 逐项目会变的"旋钮"(cut/qa 阈值、路径、说话人标签):从 json 读,不同项目
  可以有不同取值,新项目不改代码只加一份 json。
- 所有项目共享、不该变的"bug 修复"(TERMINAL 全角标点集、WAV_OUT 清空、
  EDGE_PAD 邻组裁剪、silence_qa 的多字符 token 匹配):直接写死在对应模块里,
  不放进配置——放进配置等于允许新项目"选择性关掉"一个 bug 修复。

已有 6 个项目的 json 是照抄各自当前脚本里的实际常量得到的,不是按新默认值
重新拍的:voice-lover/luozhenyu/dialogue-samples/wav-raw 四个还在用旧的
MAX_GAP=0.8 拍平压缩(gap_pass_ratio=null)+ 2-30s + FIXED_GAP_TH=1.15;
只有 kuaidao/wowan 用了新的比例压缩(gap_pass_ratio=0.3)+ 5-28s + 1.5。
迁移到这套代码本身不应该改变任何已有项目重跑的输出——差异只应该来自
"这几个项目还没被要求换新参数",而不是重构引入的意外行为变化。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = PACKAGE_ROOT / "projects"


@dataclass
class CutConfig:
    """apply_merge 阶段(cut.py)的参数,对应旧 apply_merge.py 里的常量。"""
    edge_pad: float = 0.20
    max_gap: float = 0.80
    # None = 旧的"拍平"压缩(不管超出多少,统一压到 max_gap);
    # 数字 = 新的"比例保留"压缩(方案2):
    #   target_gap = max_gap + (原始gap - max_gap) * gap_pass_ratio
    gap_pass_ratio: Optional[float] = None
    min_dur: float = 2.0
    max_dur: float = 30.0
    xfade: float = 0.02
    # 导出切片的首尾淡入淡出(秒,产品默认 60ms):源有底噪时,切片边界的
    # 硬切会产生可听的咔哒/底噪突起。头部从零平滑淡入;尾部淡出平滑到
    # 静音(尾部若是 min_tail_sil 补出来的纯零,淡出自然落在静音上无副
    # 作用)。0 = 关,仅用于历史项目 parity 复现。
    fade_in: float = 0.06
    fade_out: float = 0.06
    energy_floor_db: float = -42.0
    energy_search_s: float = 0.80
    energy_pad_s: float = 0.08
    # None = 关(旧行为,保持 parity);数字 = 每条导出音频的句尾静音保底秒数:
    # 按响度实测真实语音结束点,句尾静音不足时补零(带淡出)到该值。邻组间隙
    # 小于 2*edge_pad 时中点裁剪会把尾巴裁到只剩 gap/2(wowan 最短 ~0.05s),
    # 源音频里没有静音可延,只能补,不能往邻组地盘里扩。
    # 注意:min/max_dur 过滤在补零之前判,所以开启后 max_dur 是软上限——
    # 最终时长最多超出 min_tail_sil 秒(且超出部分是纯静音)。max_dur 如果
    # 是训练端的硬限制,配的时候自己减掉这个量。
    min_tail_sil: Optional[float] = None


@dataclass
class QaConfig:
    """silence_qa 阶段的参数,对应旧 qa_silence_outliers.py 里的常量。"""
    fixed_gap_th: float = 1.15


@dataclass
class ParaCutConfig:
    """para_cut 阶段(剪除对齐 gap 内未转写声音,如笑声/语气声)的参数。
    判定只用响度+对齐时间戳,不引入模型:gap 内能量高于 cut.energy_floor_db、
    持续超过 min_cut 的声音视为未转写内容,在静音谷落刀剪除。"""
    min_cut: float = 0.15            # 事件最短时长,短于此不值得剪
    guard: float = 0.08              # 刀口距相邻对齐字边界的最小距离
    merge_gap: float = 0.12          # 相邻事件间隔小于此合并为一刀
    min_residual_pause: float = 0.25  # 句中剪除后残余停顿下限,不足补零


@dataclass
class DenoiseConfig:
    """denoise 阶段(语音增强/降噪,可选后置)的参数。
    对 TTS 数据,降噪的红线是不伤音色:每条降噪前后各算 ECAPA 说话人向量,
    余弦相似度低于 spk_sim_th 的条目判"音色受损",--apply 时保留原音频并
    记入报告,绝不静默替换。跑在 para-cut 之后(降噪会改变能量底噪,
    para-cut 的静音谷判定要在原始响度上做)。"""
    model: str = "MossFormer2_SE_48K"   # ClearerVoice-Studio 模型名
    spk_sim_th: float = 0.90            # 音色保真下限,低于此不 apply


@dataclass
class LoudnormConfig:
    """loudnorm 阶段(音量归一化,项目内集级增益)的参数。"""
    # None = 自动取本项目各集响度的中位数为目标;数字 = 指定目标(dBFS,
    # active-speech RMS 口径)。分开训、只做项目内归一化时用 None 即可。
    target_db: Optional[float] = None
    # 集内条级偏差的保留比例。注意语义和 gap_pass_ratio 方向相反:
    # 这里 None = 完全保留集内自然强弱(只做集级增益,默认);
    # 0~1 = 集内偏差乘以该比例收敛(0 = 逐条拍平,不建议)。
    utt_pass_ratio: Optional[float] = None
    # 峰值上限:增益被 min(目标增益, 上限-峰值) 钳制,永不削波。
    peak_ceiling_db: float = -1.0


@dataclass
class Stage1Config:
    """separate+VAD+说话人聚类+ASR 阶段的参数,6 个项目里从未变过,
    但仍然放进配置而不是硬编码——新项目如果真的需要不同的 VAD/聚类
    灵敏度,不用改代码。"""
    sr: int = 44100
    vad_sr: int = 16000
    split_gap: float = 0.80
    min_seg: float = 1.0
    max_seg: float = 18.0
    spk_dist: float = 0.60
    bsr_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    asr_batch: int = 32
    # 目标说话人参考音频(该说话人几秒~几十秒的干净独白样本,wav/mp3 均可)。
    # None = 旧行为:锁定总时长最大的簇为主说话人(适合"主播就是目标"的源)。
    # 路径 = 用 ECAPA 向量在各簇中挑与参考余弦距离最近的簇——目标说话人
    # 不是说话最多的人、或不同集主次说话人会换人时,必须用这个模式。
    ref_audio: Optional[str] = None
    # 参考模式下的距离上限:最近簇距离仍超过该值,判定"本集没有目标说话人",
    # 整集不保留任何段(target_found=false 写进 report),宁缺毋滥。
    ref_max_dist: float = 0.60
    # 长段二分裂的分级候选断点(产品默认开):切 >max_seg 的段时先只考虑
    # >=0.5s 的间隙(多为句末/分句),没有再逐级放宽,保底 0.15s(宁可断
    # 逗号也不丢整段)。None = 旧行为:所有间隙同台竞争,仅用于复现历史
    # 项目的 parity 验证。
    split_tiers: Optional[list] = field(
        default_factory=lambda: [0.5, 0.3, 0.15])


@dataclass
class MergeReviewConfig:
    """prep_review.py / merge_llm.py 的贪心归组参数,6 个项目里也从未变过。"""
    max_gap: float = 2.5
    # 产品策略(默认):组硬上限 29s(给 cut 的 edge_pad+min_tail_sil 留
    # ~1s 余量,导出后 <=30s),优先区间 5-20s——打包到 pref_max_dur 为止,
    # 超出后进入 [pref_max_dur, max_dur] 弹性区,只在间隙 >= good_gap 的
    # "好断点"(句末/分句停顿)断开;到硬上限仍没有好断点,回溯到组内最大
    # 间隙处断(左半不短于 pref_min_dur)。pref_max_dur=None 关闭优先区间
    # (旧行为:直接打包到 max_dur),仅用于历史项目 parity 复现。
    max_dur: float = 29.0
    pref_max_dur: Optional[float] = 20.0
    pref_min_dur: float = 5.0
    good_gap: float = 0.5


@dataclass
class ProjectConfig:
    name: str
    src_dir: str
    out_root: str
    speaker_tag: str            # filelist.txt 里的说话人字段
    seg_prefix: str             # 导出 wav 的 id 前缀,如 "vl"/"kd"/"ww"
    file_glob: list = field(default_factory=lambda: ["*.wav"])
    # idx_mode:
    #   "prefix3"    - 文件名前 3 个字符就是集号(旧项目,文件名本身如 001.m4a)
    #   "sequential" - 按 file_glob 排序后依次编号 000/001/...(源文件名不规则时用)
    idx_mode: str = "prefix3"
    # sequential 模式下,可选按指定文件名(不含扩展名)顺序排列,而不是字典序
    episode_order: Optional[list] = None
    llm_audit_speaker_desc: str = "单说话人男声"  # 塞进 llm_audit 提示词里描述人物
    cut: CutConfig = field(default_factory=CutConfig)
    qa: QaConfig = field(default_factory=QaConfig)
    para_cut: ParaCutConfig = field(default_factory=ParaCutConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    merge_review: MergeReviewConfig = field(default_factory=MergeReviewConfig)
    loudnorm: LoudnormConfig = field(default_factory=LoudnormConfig)

    @property
    def src_path(self) -> Path:
        return Path(self.src_dir)

    @property
    def out_path(self) -> Path:
        return Path(self.out_root)

    @property
    def work_dir(self) -> Path:
        return self.out_path / "work"

    @property
    def reports_dir(self) -> Path:
        return self.work_dir / "reports"

    @property
    def merge_dir(self) -> Path:
        return self.out_path / "merge"

    @property
    def dataset_dir(self) -> Path:
        return self.out_path / "dataset"

    @property
    def wavs_dir(self) -> Path:
        return self.dataset_dir / "wavs"

    @property
    def alignments_path(self) -> Path:
        return self.out_path / "alignments.jsonl"


def _dc_from_dict(cls, d):
    if d is None:
        return cls()
    return cls(**d)


def load_project(name: str) -> ProjectConfig:
    """从 projects/<name>.json 加载配置。name 可以带 .json 后缀也可以不带。"""
    path = PROJECTS_DIR / (name if name.endswith(".json") else f"{name}.json")
    if not path.exists():
        available = sorted(p.stem for p in PROJECTS_DIR.glob("*.json"))
        raise FileNotFoundError(f"找不到项目配置 {path}。可用项目: {available}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = dict(raw)
    cut = _dc_from_dict(CutConfig, raw.pop("cut", None))
    qa = _dc_from_dict(QaConfig, raw.pop("qa", None))
    para_cut = _dc_from_dict(ParaCutConfig, raw.pop("para_cut", None))
    denoise = _dc_from_dict(DenoiseConfig, raw.pop("denoise", None))
    stage1 = _dc_from_dict(Stage1Config, raw.pop("stage1", None))
    merge_review = _dc_from_dict(MergeReviewConfig, raw.pop("merge_review", None))
    loudnorm = _dc_from_dict(LoudnormConfig, raw.pop("loudnorm", None))
    return ProjectConfig(cut=cut, qa=qa, para_cut=para_cut, denoise=denoise,
                         stage1=stage1, merge_review=merge_review,
                         loudnorm=loudnorm, **raw)


def list_projects() -> list:
    return sorted(p.stem for p in PROJECTS_DIR.glob("*.json"))
