# voice-design-pipeline

从原始长音频(可含多个说话人和背景音乐)中提取**指定目标说话人**的干净语音,
产出可直接用于 TTS 训练 / 音色定制的 `dataset/`。只做数据处理,训练在下游
框架自己的仓库里做。

每个数据源一份 json 配置,共用同一套代码;流水线由无状态的 `run.py` 逐
stage 执行,stage 之间的门禁与决策见 [AGENTS.md](AGENTS.md)(面向 AI agent
的流程手册)。

## 快速开始

```bash
bash scripts/setup_env.sh                 # 从锁文件重建 .venv 与 .venv-enhance
.venv/bin/python run.py doctor            # 环境自检;加项目名再做 60s 分离冒烟
cp projects/example.json projects/myproject.json   # 改 src_dir / out_root / speaker_tag
.venv/bin/python run.py extract myproject          # GPU:分离 + VAD + 锁定说话人 + 转写
.venv/bin/python run.py group myproject && .venv/bin/python run.py group-apply myproject
.venv/bin/python run.py export myproject           # 重切导出 dataset/
.venv/bin/python run.py align myproject            # GPU:强制对齐 + 整条重转写
.venv/bin/python run.py fix-punct myproject && .venv/bin/python run.py fix-tail-punct myproject
.venv/bin/python run.py remove-nonspeech myproject --apply && .venv/bin/python run.py align myproject
.venv/bin/python run.py qa-silence myproject dataset   # 报告;确认后再加 --apply
.venv/bin/python run.py qa-speaker myproject
.venv/bin/python -m pytest tests -q               # 改过 ttspipe/ 后必跑,秒级
```

`run.py all myproject` 一把跑完 extract → fix-punct;需人工复核的阶段不自动跑。

## 流水线一览

| 阶段 | 做什么 | 备注 |
|---|---|---|
| `doctor` | 依赖 / CUDA / 侧车 venv 自检,可选 60s BS-RoFormer 分离冒烟 | 新音源必做 |
| `extract` | BS-RoFormer 去 BGM → Silero VAD 分段 → ECAPA 聚类锁定目标说话人 → Qwen3-ASR 转写 | GPU;每集一份 `work/reports/report_<idx>.json`,已存在则跳过 |
| `group` / `group-llm` / `group-apply` | 贪心归组(产品时长策略)→ 可选 LLM 语义复审 → 应用并校验 | 不跑 LLM 时贪心结果直接生效 |
| `export` | 按组重切:压缩组内长停顿、邻组 pad 互不越界、60ms 淡入淡出、句尾静音保底 | **重跑清空 dataset/wavs**;写 `manifest.json` |
| `align` | 对每条导出音频整条重转写 + 字级时间戳 → `alignments.jsonl` | GPU;按 id 断点续跑 |
| `fix-punct` | 采纳重转写为权威文本,按实测停顿删假逗号 / 补漏逗号 | 文本-时间戳用 difflib 对齐 |
| `fix-tail-punct` | 被时长上限强制断开的条目,结尾"。"如实改"," | 读 manifest 定位组边界 |
| `remove-nonspeech` | 剪掉对齐 gap 里未转写的笑声 / 语气声(响度 + 对齐驱动,无模型) | 静音谷落刀、剪后复测;`--apply` 后需重跑 `align` |
| `qa-silence` | 首尾静音 IQR + 句中停顿固定阈值,标记异常条目 | `--apply` 剔除 |
| `qa-speaker` | 2s 滑窗 ECAPA 相似度找混入的他人声音,分重度 / 轻度 | `--apply` 仅剔重度,wav 移入可恢复目录 |
| `denoise` | MossFormer2_SE_48K 语音增强,逐条 ECAPA 音色验收 | 用 `.venv-enhance` 跑;低于阈值保留原音频 |
| `loudnorm` | 项目内集级增益拉平集间响度,峰值钳制不削波 | `--apply` 就地改写,增益记录在报告里 |
| `qa-llm` / `qa-prosody` | LLM 打分 + 排除项 / 声学韵律指标 | 可选 |

旧名(`stage1` `merge-prep` `merge-llm` `merge-apply` `cut` `punct-fix`
`tail-punct` `para-cut` `silence-qa` `llm-audit` `prosody`)仍是有效别名。

所有带 `--apply` 的阶段默认只出报告,报告先行、确认后再改数据。

## 关键策略

**时长(产品默认)**:单条硬上限 30s,优先区间 5-20s。组打包到 20s 后进入
20~29s 弹性区,只在 ≥0.5s 的句末级停顿断开;到硬上限仍没有好断点,回溯到
组内最大间隙。extract 切超长段用分级断点(≥0.5s → ≥0.3s → ≥0.15s 保底),
两者共同把断点推向句末。`pref_max_dur=null` / `split_tiers=null` 退回旧的
"直接打包到上限"行为,仅用于历史项目复现。

**目标说话人**:默认锁定总时长最大的簇;`stage1.ref_audio` 给参考样本时改为
选与参考最近的簇,最近距离超过 `ref_max_dist` 判该集无目标说话人、整集丢弃
(宁缺毋滥)。多簇时自动导出各簇试听预览(`work/spk_preview/`),人工指认后
把预览 wav 填回 `ref_audio` 重跑。段级聚类拦不住段内几秒的他人插话,由
`qa-speaker` 窗级检测补网。

**文本权威**:align 对整条音频的重新转写(有完整上下文)替换 export 的段级
拼接;与旧文本相似度 <0.5 的条目视为转写异常保留旧文本。

**停顿与静音**:组内间隙超过 `max_gap` 才压缩,`gap_pass_ratio` 让长停顿按
比例保留差异而不是拍平;压缩边界按响度回退而非 VAD 声明的边界;`qa.fixed_gap_th`
必须与 `cut.max_gap` / `gap_pass_ratio` 配套改。

**响度**:响度差异的大头在集与集之间,所以 loudnorm 默认只做集级增益,集内
强弱保留;`utt_pass_ratio` 生效时二次 apply 不幂等,撤销靠报告里的 `applied_gains`。

## 数据一致性

- **manifest**:`dataset/manifest.json` 记录每条音频来自哪一集、哪个归组、
  哪些源段、源音频时间范围——可追溯、可从源重切复原,下游按构造对应而不猜。
- **provenance**:`export` / `align` 写产物时记录输入指纹
  (`out_root/provenance/`),`fix-punct` / `fix-tail-punct` / `remove-nonspeech` /
  `qa-silence` 开跑前校验,过期就拒跑并说明该重跑哪个 stage。指纹按每条音频
  的采样数而非字节,所以 denoise / loudnorm / 文本改写不会误报,重切和剪除会。
- **写死的修复(不放进配置)**:TERMINAL 标点集含全角+半角(否则 join 出
  "，,");重切前清空 `dataset/wavs` 并删除过期的 `alignments.jsonl` 与
  `*_prepunct` 备份;邻组 `edge_pad` 裁到真实边界中点;silence_qa 的 token 级
  匹配;extract 人声轨静音即报错。

## 配置

`projects/<name>.json` 一份,模板 `projects/example.json` 含全部字段与产品
默认值;含本地路径的真实配置不入库。常调的旋钮:

| 场景 | `cut.max_gap` | `gap_pass_ratio` | `cut.min/max_dur` | `qa.fixed_gap_th` |
|---|---|---|---|---|
| 朗读 / 有声书 | 0.80 | null(拍平) | 3 / 30 | 1.15 |
| 讲课 / 对话(保留停顿层次) | 1.20 | 0.3 | 3 / 30 | 1.50 |

改硬上限时按余量原则换算:`cut.max_dur` = 硬上限,`merge_review.max_dur` =
硬上限 − 1(给 edge_pad + 句尾补零留空间)。

## 环境

两个 uv venv,锁文件在 `requirements/`:

- `.venv`(主):extract ~ qa-speaker 全流程。
- `.venv-enhance`:clearvoice(denoise)+ funasr。**必须隔离**——这两个包会
  降级 numpy / torchaudio,曾把 BS-RoFormer 分离静默弄坏(人声轨全零);
  `doctor` 的分离冒烟就是为拦这个。

`tests/`:纯函数单测 + 合成音频端到端(report → group → export),不碰模型。

## 目录

```
run.py                 统一 CLI
ttspipe/
  config.py            ProjectConfig 与各阶段参数
  stage1.py            extract(四步在一个 Runner 里,模型只加载一次)
  merge_prep.py / merge_llm.py / merge_apply.py   group 三步
  cut.py               export
  align.py / punct_fix.py / tail_punct.py / textalign.py
  para_cut.py          remove-nonspeech
  silence_qa.py / speaker_qa.py / denoise.py / loudnorm.py / llm_audit.py / prosody.py
  provenance.py / doctor.py / gpu.py
projects/example.json  配置模板
requirements/          两个 venv 的锁文件
scripts/setup_env.sh   重建环境
tests/
```
