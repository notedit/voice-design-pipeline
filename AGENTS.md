# AGENTS.md — voice-design-pipeline

本文件是 agent 的**流程控制中枢**:各 stage 何时跑、跑完检查什么、门禁不过
怎么办,都以这里为准。`run.py` 只是无状态执行器,stage 之间的编排与决策
由 agent 按本文件执行。

## 使命

输入一段(或一批)音频——可能含多个说话人和背景音乐——提取**指定目标
说话人**的干净语音,产出可直接用于该说话人 TTS 训练的
`dataset/{wavs,metadata.csv,filelist.txt}`。

**边界:本项目只做数据处理,到交付 dataset + QA 报告为止。**模型微调/
训练是下游环节,在训练框架自己的仓库里做,不纳入本流水线,也不要往
这个 repo 里加训练相关的代码或脚本。

两条硬原则:
1. **宁缺毋滥**:混入其他说话人/残留 BGM 的数据比少量数据危害更大,
   拿不准的段宁可丢弃。
2. **报告先行**:所有带 `--apply` 的操作(剔除条目/改写 wav)先出报告给
   用户看,用户确认后才 apply。

## 第 0 步:环境自检(每次会话开始先做)

```bash
.venv/bin/python run.py doctor            # 依赖/CUDA/侧车 venv
.venv/bin/python run.py doctor <project>  # + 60s BS-RoFormer 分离冒烟(新音源必做)
.venv/bin/python -m pytest tests -q       # 改过 ttspipe/ 后必跑,秒级
```
- 环境从锁文件重建:`bash scripts/setup_env.sh`(requirements/*.lock.txt)。

- extract/align 需要 GPU + 上述全部依赖;group/export/qa-silence/
  loudnorm/qa-prosody 只需 numpy/librosa/soundfile/sklearn。
- **双 venv 隔离(2026-08 事故后的铁律)**:仓库根下有两个 uv venv——
  - `.venv`(主):extract~remove-nonspeech 全流程依赖(torch/silero-vad/
    speechbrain/qwen-asr/audio-separator/librosa/soundfile/pyworld/
    sklearn/onnxruntime/audioread)。**严禁往里装 clearvoice/funasr 等
    额外 ML 包**:它们会降级 numpy/torchaudio,曾把 BS-RoFormer 分离
    静默搞坏(人声轨全零、流水线安静地产出空数据集)。stage1 已加硬
    校验,人声轨静音会直接报错。
  - `.venv-enhance`:clearvoice(denoise 阶段)+ funasr(SenseVoice
    实验)+ speechbrain。denoise 阶段用
    `.venv-enhance/bin/python run.py denoise ...` 跑。
  - 新增依赖前先判断归属:属于主流程的进 `.venv`,任何"顺手装个模型
    试试"的进 `.venv-enhance` 或临时 venv,装完必须重验
    BS-RoFormer 分离(60s 样本,人声轨应有能量)。
- LLM stage(`group-llm` / `qa-llm`):
  - 在 Claude Code 里跑时,**不要走 API 脚本**,直接用 Claude Code 的
    subagent 完成语义归组/质检判定,模型尽量选 sonnet 或 haiku(控制成本,
    这两类任务不需要最强模型)。输入输出沿用脚本的文件格式
    (`merge/input_<idx>.json` -> 语义拆分结果;审计结果 schema 同
    `llm_audit.py`),这样下游 stage 不感知差异。
  - 独立脚本方式(`ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY`)只在
    Claude Code 之外运行时用;没有 key 就跳过这两个可选 stage,不要
    索要或猜测 key。
- GPU 由 `ttspipe/gpu.py` 自动挑,不要手动设 `CUDA_VISIBLE_DEVICES`。

## 第 1 步:建项目

1. 把源音频放进一个目录(单文件也建目录)。
2. 抄 `projects/example.json` 为 `projects/<name>.json`:`src_dir` 指向该
   目录,`out_root` 指向输出位置,`file_glob` 覆盖实际扩展名,文件名不是
   `001.xxx` 格式就用 `"idx_mode": "sequential"`。
3. **目标说话人**:
   - 用户给了目标说话人参考样本(几秒~几十秒干净独白)→ 填进
     `stage1.ref_audio`。
   - 没给,且音频明显是单人主导(讲课/有声书)→ `ref_audio: null`
     (选总时长最大簇)。
   - 没给,且是多人对话/目标未必说话最多 → 先按 null 跑 extract,走下面
     extract 门禁里的"人工指认"分支。
4. 真实配置含本地路径,不入库(.gitignore 已配好)。

## 第 2 步:逐 stage 执行与门禁

统一命令形式:`python run.py <stage> <project> [参数] [--apply]`。
按顺序推进,每个 stage 过了门禁才进下一个;`all` 只在门禁逻辑全部
自动可判时才用,首次处理新音源一律逐 stage 跑。

标准顺序:`extract -> group -> [group-llm] -> group-apply -> export ->
align -> fix-punct -> fix-tail-punct(agent 自主,见下)->
remove-nonspeech(agent 自主,检出即 --apply + 增量重 align)->
qa-silence -> qa-speaker(agent 自主跑报告)-> [denoise] -> [loudnorm] ->
[qa-llm/qa-prosody]`。
旧名(stage1/merge-prep/merge-llm/merge-apply/cut/punct-fix/tail-punct/
para-cut/silence-qa/llm-audit/prosody)仍是有效别名,新文档一律用新名。

### extract(BGM 分离 + VAD + 锁定说话人 + ASR;GPU,长任务,后台跑;旧名 stage1)

- 产物:`out_root/work/reports/report_<idx>.json`(每集一份)。
- 断点续跑:report 已存在则跳过;**换 ref_audio/参数后必须删掉对应
  report 再跑**,否则改动不生效。
- 门禁(逐集看 report):
  - `target_found` 为 false → 该集判定无目标说话人。核对 `spk_clusters`
    里各簇的 `ref_dist`:全都远(>0.7)说明说话人确实不在或参考样本太差;
    刚好卡在 `ref_max_dist` 附近(0.6~0.7)则把距离数字报给用户,由用户
    决定放宽 `ref_max_dist` 还是换更长/更干净的参考样本重跑。
  - `n_clusters > 1` 且没配 `ref_audio` → **停下走人工指认**:把
    `work/spk_preview/<idx>/cluster_<k>.wav` 各簇预览和 `spk_clusters`
    统计(段数/时长)报给用户听并指认;把选中的预览 wav 填进
    `stage1.ref_audio`,删 report 重跑本集。没有人工确认不许默认
    "最长簇就是目标"继续往下走。
  - kept 总时长过小(单集 <60s 或全部集合计 <10min)→ 提醒用户数据量
    可能不足以训练,问是否继续。
  - `dropped_asr` 占比 >20% → BGM 分离或音质可能有问题,抽 2~3 段
    kept=false 的原文报给用户核对。

### group(贪心归组基线;CPU,秒级;旧名 merge-prep)

- 产物:`merge/input_<idx>.json` + 贪心分组。
- 门禁:无,直接进下一步。
- 时长策略(**产品默认**):单条硬上限 30s、优先区间 5-20s——
  `merge_review` 默认 `max_dur=29.0`(给 cut 的 pad 留 ~1s 余量)+
  `pref_max_dur=20.0` + `good_gap=0.5`,配合 `stage1.split_tiers`
  `[0.5,0.3,0.15]` 分级断点,断点优先落句末。新项目不用配,直接吃默认;
  用户指定别的区间时按同样的余量原则换算(cut.max_dur = 硬上限,
  merge max_dur = 硬上限 - 1)。
  - `pref_max_dur=null` / `split_tiers=null` = 关闭,退回旧行为,仅用于
    历史项目 parity 复现。
  - 处理完看 fix-tail-punct 报告的 forced_boundary 占比,验证断点质量。

### group-llm(LLM 语义归组;可选;旧名 merge-llm)

- 用户明确要求语义归组时才做;跳过时贪心基线直接生效,流水线不卡。
- 在 Claude Code 里用 subagent(sonnet/haiku)读 `merge/input_<idx>.json`
  做语义拆分,产物写回 `merge_llm.py` 相同的输出格式;不走 API 脚本。

### group-apply(应用拆分 + 校验;旧名 merge-apply)

- 产物:`merge/groups_<idx>.json`。
- 门禁:内置 `validate_groups` 校验失败会报错——报错时不要绕过校验,
  把错误交给用户。

### export(重切导出;CPU;旧名 cut)

- 产物:`dataset/wavs/*.wav` + `metadata.csv` + `filelist.txt`。
- **破坏性**:重跑会清空重建 `dataset/wavs/`。目录已存在且非本次会话
  产出时,先确认用户要覆盖。
- 门禁:导出条数/总时长与 stage1 kept 时长量级一致(归组+时长过滤会
  损失一部分,断崖式缩水不正常);抽 3 条 wav 时长和 metadata 对得上。
- **切完必做:给出 qa-speaker 处理意见**(判断规则见 qa-speaker 小节的
  决策表),连同依据(各集簇数、内容类型)一起报给用户;判定"建议跑"
  时直接自主跑报告,不等确认。

### align(强制对齐;GPU)+ fix-punct(标点校正;旧名 punct-fix)

- 产物:`alignments.jsonl`;punct-fix 就地改写 metadata 文本。
- punct-fix 的 `kept_old`(匹配失败回退)比例高是已知限制,不算故障,
  数字如实报告即可。

### fix-tail-punct(尾标点如实化;agent 自主;旧名 tail-punct)

- 背景:归组被 max_dur 强制断开时(边界停顿只有逗号级),ASR 打的段尾
  "。"不反映真实语调——音频是非终止语调、文本却是句号,训练会学错句尾
  韵律映射。语速密的讲课/对话源强制断开占比可到 70%+。
- agent 在 fix-punct 后自主跑 `run.py fix-tail-punct <project>`:按
  merge/input 里的边界停顿,把强制断开(间隙 <0.8s)条目的结尾"。"改
  ","(问号/感叹号/引号收尾不动)。纯文本操作、有报告可回溯,自主执行。
- 归组与 metadata 的对应是顺序模糊匹配,对应失败会整体报错不改——报错
  说明 merge/ 与 dataset/ 不是同代产物,先查再跑。

### remove-nonspeech(剪除未转写的副语言声音;**agent 自主决策,优先执行**;旧名 para-cut)

- 目的:对齐 gap 内能量高于底噪、时长 ≥ min_cut 的声音(笑声/语气声/
  感叹)= 文本里没有的内容,剪掉后文本-音频一致。判定只用响度+对齐
  时间戳,无模型依赖。
- **自主执行,不等用户确认**:align 完成后先跑报告模式(秒级、零改动),
  只要检出可安全剪除的事件(edited > 0 且 verify 全过),就直接
  `--apply` → 增量重跑 `align` → 再做后续 QA。它排在 qa-silence/loudnorm
  之前——不先剪,qa-silence 的"假静音"标记全是噪声。可以自主的依据:
  判定保守(guard + 静音谷 + 剪后复测失败自动回退)、文本零改动、
  dataset 可随时从源音频重切复原,不满足"破坏性操作需确认"的条件。
- 安全规则(写死在实现里):刀口距对齐字边界 ≥ guard;事件两端必须有
  静音谷,没有(声音与语音连续)就放弃剪除记入 skipped;剪后逐条复测,
  失败自动回退。**边笑边说不在本阶段能力内**,由 llm-audit 排除项兜底。
- **A/B 对比试听页仍然要出**(artifact,红标剪除区间、绿标接缝),但作为
  事后复核交付物随汇报附上,不是前置门禁。
- 仍留给用户的决定:below_min_dur 条目、no_valley 残留(剪不掉的贴字
  笑声)是否整条剔除——这些会删数据,不自主执行。

### qa-silence / loudnorm / qa-prosody(QA 三件套;默认只出报告;旧名 silence-qa/prosody)

- 顺序:先 qa-silence(剔除异常条目)再 loudnorm(改写响度),prosody
  可选。
- 都先跑无 `--apply` 的报告,把 flagged 条数/分布摘要报给用户,**用户
  确认后**才加 `--apply`。
- **试听预览(qa-silence 报告出来后必做)**:从 flagged 里按标记时长挑
  最长的 ~10 条,生成网页预览发布为 artifact 给用户复核——每条带可点击
  定位的音频波形、问题区间高亮(head/tail/gap 的起止时间从 alignments
  按 silence_qa 的 match 逻辑换算)、转写文本;音频转 64kbps mp3 以
  data URI 内嵌,页面控制在 16MB 内。
- 复核前先用响度实测校验报告口径:qa-silence 的 head/tail 是对齐口径
  (最后转写字到文件尾),"长静音"里可能是未转写的笑声/语气声(用 ASR
  单独转写该区间即可确认)——两者的处置结论不同,报告给用户时要区分。
- loudnorm `--apply` 就地改写 wav,撤销只能靠报告里的 applied_gains;
  `utt_pass_ratio` 生效时二次 apply 不幂等,严禁重复 apply。
- `qa.fixed_gap_th` 必须与 `cut.max_gap`/`gap_pass_ratio` 配套改。

### qa-speaker(说话人纯度验证;agent 自主决策+跑报告,剔除需用户确认)

- 背景:extract 的聚类是**段级**的,一段里混入几秒他人声音(主持人插话、
  演示播放的音频、观众声),整段向量仍被主说话人主导而并入主簇——段级
  和整条口径都看不出来,必须窗级检测。这是 kuaidao kd000_0001 实际踩过
  的漏网案例。
- **是否要跑,由 agent 在 export 完成后按此表给意见**(报告模式只依赖
  dataset/wavs,不需要 alignments,可以立即跑):

  | 信号 | 意见 |
  |---|---|
  | extract 有任何一集聚出 >1 簇(源里确有其他声音) | **跑**(被丢弃簇的邻接段边界大概率有沾染) |
  | 全部单簇,但内容是讲课/对话/访谈等有互动的类型 | **跑**(kuaidao 案例证明:单簇 ≠ 纯净,段内混入会被聚类吞进主簇) |
  | 全部单簇 + 纯朗读/有声书(录音棚单人朗读) | 可跳过;用户要求最高纯度时仍可跑 |

  成本参考:GPU 上约 2-3 分钟 / 100 条,倾向跑而不是省。
- 做法(`ttspipe/speaker_qa.py`):稳健质心 + 2s 滑窗 ECAPA 相似度;
  连续 ≥2 个低窗(<0.45)标记;连续 ≥4 窗或最低 ≤0.05 判**重度**
  (基本确定混入),其余**轻度**(演示音频/远场人声误报率高)。
- agent 在 qa-silence 后自主跑报告,**必出试听复核页**(artifact:低窗
  区间红色高亮 + 可定位播放,重度/轻度分组)。`--apply` 只剔除重度
  (wav 移入 work/speaker_qa_removed/ 可恢复),且需用户确认;轻度
  一律等用户听完逐条定夺。

- 基于 ClearerVoice-Studio(默认 `MossFormer2_SE_48K`),跑在 dataset/wavs
  上。**报告模式由 agent 自主跑**(零改动,GPU 分钟级),跑完按下面的
  规则自动给出建议,连同关键数字一起报给用户:

  | 报告指标 | 建议 |
  |---|---|
  | sim 中位 <0.98 或 low_sim 条目 >5% | **不建议 apply**,先人工抽听最差条目 |
  | floor_gain 中位 ≥6dB 且音色全过验收 | **建议 apply**(底噪明显,收益大) |
  | floor_gain 中位 <3dB 但 ≥10% 条目 gain ≥6dB,音色全过 | **建议 apply**(整体干净但有局部噪段,无音色代价) |
  | floor_gain 中位 <3dB 且高增益条目 <10% | **建议不 apply**(没什么可降的) |

- **音色保真红线**(写死在实现里):每条降噪前后各算 ECAPA 说话人向量,
  余弦相似度低于 `denoise.spk_sim_th`(默认 0.90)的条目判音色受损,
  --apply 时保留原音频,绝不静默替换。
- 顺序在 remove-nonspeech 之后(para-cut 的静音谷判定要在原始底噪上做);逐采样
  对齐、不改时间结构,alignments 保持有效,不需要重跑 align。
- 预览在 work/denoise_preview/;**--apply 仍需用户确认**——agent 给建议
  不代行决定,因为"要不要接受音质被模型加工过"是数据口味问题。

### qa-llm(LLM 质检;可选;旧名 llm-audit)

- 用户要求才跑;跑之前确认 `llm_audit_speaker_desc` 与实际说话人相符。
- 在 Claude Code 里用 subagent(sonnet/haiku)按 `llm_audit.py` 的 schema
  (韵律分+自然度分+四项排除标准)逐条判定并写相同格式的结果;条数多时
  可并行多个 subagent 分片处理。不走 API 脚本。

### 完成汇报

最终交付时报:总条数、总时长、时长分布(min/median/max)、被 QA 剔除
的条数及原因分布、`target_found=false` 的集列表、dataset 路径。

## 改代码约束(开发型任务)

1. `ttspipe/` 是所有项目共用的,算法改动影响每个项目重跑的结果;逐项目
   取舍走 `projects/<name>.json`,不许在代码里写死某项目的特殊逻辑。
2. 写死的 bug 修复不许配置化(TERMINAL 全角+半角标点集、cut 前清空
   WAV_OUT、cut 时删除过期 alignments.jsonl、edge_pad 邻组裁剪、
   silence_qa 的 token 级匹配)。
3. stage1 是一个整体(`Stage1Runner`),不要拆成独立模块——拆开每步都要
   重新加载 GB 级模型,是已经做过并否决的方案。
4. 文本权威 = align 对整条音频的重新转写;punct_fix 用 `textalign.
   match_chars`(difflib)对时间戳,改匹配逻辑先跑 tests 再用真项目比
   kept_old 数字。
5. **产物新鲜度靠 provenance 校验,不靠肉眼**:写新 stage 时,产物被下游
   消费的要 `stamp()`,消费上游产物的要 `require_fresh()`(见
   `ttspipe/provenance.py`);拒跑报错说明该重跑哪个 stage,照做,不要绕。
6. 验证方式是 parity test:改动后挑有产出的项目重跑该阶段,与现网产物
   对比;预期无行为变化必须逐字节一致,预期有变化的差异必须逐条可归因。
   `select_target_cluster`(stage1.py)是纯函数,选簇逻辑的改动先用假
   向量写脚本测过再上真数据。

## 文档与保密约定

- 对外文档(README、注释、提交信息)不写具体数据来源:不出现上游站点
  名、具体项目/人名,统一用"数据源/`myproject`"指代。含真实路径与名字
  的配置 json 一律不入库。
- 流水线行为变更后同步更新 README 对应章节和本文件的门禁描述。
