# tts-pipeline

missevan 系 TTS 数据流水线的统一实现。之前 voice-lover-tts / luozhenyu-tts /
kuaidao-tts / wowan-tts / dialogue-samples-tts / wav-raw-tts 六个项目各自维护
一份几乎相同的脚本(pipeline.py / apply_merge.py / align_all.py / ...),改动
只发生在需要的项目上,其余的原地不动——这份重构把"从没分叉过的部分"抽成
共享代码,把"真正分叉过的部分"认真设计并保留可配置的空间,不建议再复制
整个项目目录来接入新数据源。

## 现状调研(重构前先量的,不是拍脑袋)

对比六个项目的脚本发现:

- `pipeline.py`(分离+VAD+说话人聚类+ASR)、`prep_review.py`、`merge_llm.py`、
  `apply_splits.py`、`validate_groups.py`、`align_all.py`、`punct_fix.py`
  **六个模块字节级相同**(除了路径和说话人标签字符串),六个项目从没让它们
  分叉过。
- 只有 `apply_merge.py` 和 `qa_silence_outliers.py` **真正分叉了**——
  kuaidao/wowan 在本次会话里踩了一连串坑,改了三处参数、修了两个 bug,
  另外四个项目还停在最初的版本上。
- `qa_silence_outliers.py` 只有 voice-lover/kuaidao/wowan 三个项目有,另外
  三个从没做过这层 QA。
- `llm_audit*.py` 有四代:voice-lover 的 `_all`/`_strict`/`_v4` 判定标准
  各不相同,kuaidao/wowan 的 `llm_audit.py` 是最新、schema 最完整的一版
  (韵律分+自然度分+四项排除标准)。

这个分布形状决定了重构的做法:**大部分模块是机械搬运,只有两个模块需要
真正的设计**。

## 目录结构

```
tts-pipeline/
  ttspipe/
    config.py       ProjectConfig:每个数据源一份 json,见 projects/
    gpu.py           挑空闲显存最多的 GPU
    stage1.py        分离 BGM -> VAD 分段 -> ECAPA 聚类锁定主说话人 -> Qwen3-ASR
    merge_prep.py    report -> 归组输入 + 贪心归组基线
    merge_llm.py     [可选] LLM 语义归组(API)
    merge_apply.py   应用语义拆分(如果有)+ 校验
    cut.py           按 groups 重切音频,压缩组内静音,产出 dataset/
    align.py         全量强制对齐 -> alignments.jsonl
    punct_fix.py     按实测停顿校正标点
    silence_qa.py    静音异常检测(IQR + 固定阈值),可选剔除
    loudnorm.py      音量归一化(项目内集级增益),可选就地改写
    llm_audit.py     [可选] inkling-small 打分+判定
    prosody.py       [可选] 声学韵律指标
  projects/
    example.json     配置模板(全字段+默认值);各数据源自己的
                     <name>.json 含本地绝对路径,不入库(见 .gitignore)
  run.py             统一 CLI
```

`stage1.py` 没有像其余模块一样按"步骤"拆成独立可调用单元(分离/VAD/聚类/ASR
各一个模块)——那样每一步都要重新加载一遍 GB 级模型,现实中也从来是当一个
整体跑的,拆开是伪模块化。四个步骤是同一个 `Stage1Runner` 里的方法,模型
只加载一次,跑完整批集数。

## 各阶段设计要点

**stage1(分离 + VAD + 说话人聚类 + ASR)**
BS-RoFormer 去 BGM -> Silero VAD 切段(`min_silence_duration_ms=150`,
`speech_pad_ms=60`)-> 间隙 ≤0.8s 的相邻段直接合并、超过 `MAX_SEG=18s` 的段
从内部最靠近中点又最大的间隙处二分裂开 -> ECAPA 说话人向量 + 层次聚类
(`distance_threshold=0.6`)锁定总时长最大的簇为"主说话人",其余全部丢弃
-> Qwen3-ASR 批量转写,转写失败(<2 字或不含中文)的段标记丢弃。
这四步的参数六个项目从没变过,配置里仍然放出来(`ProjectConfig.stage1`)
是为了让新项目在真的需要不同灵敏度时不用改代码,不代表鼓励每个项目都调。

**归组(merge_prep + merge_llm + merge_apply)**
贪心基线:相邻段间隙 ≤2.5s 且合并后总时长 ≤30s 就并进同一组,否则断开。
LLM 语义归组是贪心基线之上的可选复审——不跑的话贪心基线直接生效,流水线
不会卡住。`validate_groups` 保证最终分组覆盖完整、合并条件满足、时长不超限。

**cut(按组重切音频,`ttspipe/cut.py`)**——本次重构里真正做了设计的模块
- 组内静音压缩:间隙超过 `max_gap` 才压缩。`gap_pass_ratio=None` 时压平到
  固定 `max_gap`(旧四个项目的行为);给一个 0~1 的比例时,超出部分按比例
  保留差异(`target = max_gap + (gap-max_gap)*ratio`),原始间隙越长压缩后
  还是越长,只是打折——不再让所有长停顿在最终数据里长度一样。这是
  kuaidao 在"TTS 学出的句间停顿千篇一律"这个问题上定位到的根因:旧的拍平
  压缩把 15.5% 有真实长短变化的自然停顿全部拍进同一个窄区间。
- 压缩边界用响度回退(`energy_trim_end/start`),不是照搬 VAD 声明的边界——
  VAD 判"有没有人声活动"不是判响度,句尾自然渐弱时 VAD 边界常比响度真正
  跌破底噪晚 0.3-0.6s,不回退会导致导出音频里出现异常长的连续静音。
- 相邻两组的首尾 `edge_pad` 会互相裁到真实边界中点,不会越界到对方的
  真实内容里——这是 wowan(两人快节奏对话)暴露出来的 bug:两句话之间
  常常只隔 0.1~0.3s,固定 pad 0.2s 会把下一组开头的话录进这一条尾巴。
  wowan 208 条里 39 处受影响;kuaidao 因为长句常被 28s 上限强制切断,
  受影响的边界对占到 55%,比例比表面上"讲课语速慢"给人的印象大得多。
- 句尾静音保底(`min_tail_sil`,2026-08 新增):每条导出音频按响度实测
  真实语音结束点,句尾静音不足 `min_tail_sil` 秒时补零(带 20ms 淡出)
  到该值。补零而不是往源音频里扩,因为上一条的中点裁剪正是为了不吃到
  邻组内容——间隙 <2*edge_pad 时源音频里本来就没有足够的静音可取。
  实测现有数据集句尾 <300ms 的比例:kuaidao 54%、voice-lover 33%、
  wowan 21%、luozhenyu 17%,不是个别现象。测量帧从尾部对齐(残余采样
  落在头部),从头部对齐会把"说话持续到文件末尾"的最后不足一帧误当
  静音,补零量算少。None = 关(保持 parity);六个项目 json 目前都配了
  0.30,只在下次重切时生效,已有 dataset 不受影响。

**silence_qa(静音异常检测,`ttspipe/silence_qa.py`)**——同样做了设计
- 句首/句尾静音用 IQR(Q3+1.5*IQR)定异常边界,数据驱动,每个项目按自己
  的分布重新算。
- 句中/句末停顿(按夹在中间的标点分类)用固定阈值 `fixed_gap_th`——这组
  绝大多数值接近 0,IQR 在这种分布上不适用。这个阈值必须跟 cut 阶段的
  `max_gap`/`gap_pass_ratio` 配套:kuaidao/wowan 把压缩放宽后,合理停顿能
  到 ~1.66s,阈值也从 1.15 松到 1.5,两边不能只改一边。
- 文本-时间戳匹配是 token 级指针匹配,不是逐字符——强制对齐对英文/数字串
  (比如"SUV")会整体输出成一个多字符 item,逐字符比较会让这类 item 永远
  匹配不上,把它前面的字误判成"最后一个对齐字",拼出虚假的句尾静音。

**punct_fix(标点校正)**——有意不修的已知限制
文本匹配是逐字符顺序指针 walk,没有重新同步能力:新旧文本在靠前的位置
一旦有一处标点/字符对不上,后面全部错位,覆盖率跌破 80% 阈值就整条退回
旧文本、不做任何校正。长句(20s+)+ 口语化内容(重复、自我修正)命中率
明显更高——kuaidao 348 条里 191 条因此被跳过。这次重构只是照搬,没有改
匹配算法本身,因为修复会改变所有项目的输出,不是纯粹的模块化;需要的话
应该单独立项,换成真正的 diff/对齐算法。

**loudnorm(音量归一化,`ttspipe/loudnorm.py`,2026-08 新增)**
实测(active-speech RMS 口径)发现响度不一致的大头在"集与集之间",不在
集内:voice-lover 144 集的集中位响度横跨 20.3 dB,luozhenyu 六集横跨
17.7 dB,而集内 p5-p95 只有 5-10 dB。集间差异是录音/上传增益不一致,
放心拉平;集内波动含说话人真实强弱,逐条拍平会重蹈 gap_pass_ratio 的
教训。所以:
- 默认只做集级增益,目标 = 本项目各集响度中位数(各项目分开训,不需要
  跨项目统一水平;真要指定用 `loudnorm.target_db`)。集内偏差想收敛用
  `utt_pass_ratio`(注意语义和 gap_pass_ratio 方向相反:这里 None=完全
  保留,0=拍平)。
- 做成独立后置阶段跑在 dataset/wavs 上,不塞进 stage1/cut 前面——cut 的
  `energy_floor_db` 和 VAD 读的是绝对 dB,前置归一化会打乱 parity 验证过
  的输出;后置还能直接处理旧项目现有数据集,不用重切。
- 增益被峰值余量钳制(`peak_ceiling_db`,默认 -1 dBFS),永不削波;到不了
  目标的集在报告里逐集写明 n_clamped/achieved_db,不做静默截断(luozhenyu
  vl001/vl002 crest 太大,只能从 -37 提到 ~-32,到不了 -24 的目标)。
- `--apply` 就地改写 wav,逐条实际增益记录在报告 applied_gains 里——旧项目
  wav 重切复现不出来,这份记录是唯一撤销手段。已知限制:utt_pass_ratio
  生效时二次 apply 不幂等(偏差再乘一次 ratio),集级模式天然幂等。

**llm_audit / prosody(可选的两条尾巴)**
`llm_audit.py` 收敛到 kuaidao/wowan 的 schema(韵律分+自然度分+四项排除
标准:多说话人/爆音/异常停顿/口齿不清),人物描述从
`ProjectConfig.llm_audit_speaker_desc` 读——kuaidao 是单人男声讲课,wowan
是两人对话,voice-lover 是单人男声,luozhenyu/dialogue-samples/wav-raw 是
单人女声有声书朗读,写死在 prompt 里会答非所问。`prosody.py` 是纯声学的
韵律变化指标(F0/能量/停顿的 z-score),和 `llm_audit` 的主观打分是两套
不同的东西,字段名分别是 `acoustic_prosody_score` 和 `llm_prosody_score`,
不要合并成一个。

## 全项目共享的 bug 修复(写死,不放进配置)

这几处是修复,不是设计取舍,所有项目一视同仁,新项目不用惦记要不要选:

1. **TERMINAL 标点集须同时收全角+半角**(`cut.py`)。旧代码只收了半角
   标点和几个只有全角写法的符号,中文文本几乎全用全角标点,导致
   `join_texts()` 误判"已有标点结尾",叠加插入多余逗号(产生"，,"这种
   双标点)。voice-lover 4720 条里 879 条(18.6%)命中这个 bug,kuaidao
   348 条里 13 条,规模比最初以为的大得多。
2. **重跑前清空 `WAV_OUT`**(`cut.py`)。归组变化会让 seg_id 编号挪位,
   不清空会在目录里留下不属于当前 metadata.csv 的孤立 wav。
3. **`edge_pad` 邻组裁剪**(`cut.py`),见上面 cut 阶段的说明。
4. **token 级指针匹配**(`silence_qa.py`),见上面 silence_qa 阶段的说明。

## 配置里保留的旋钮(逐项目不同,是设计取舍不是 bug)

| 项目 | max_gap | gap_pass_ratio | min/max_dur | fixed_gap_th |
|---|---|---|---|---|
| voice-lover | 0.80 | null(拍平) | 2.0 / 30.0 | 1.15 |
| luozhenyu | 0.80 | null | 2.0 / 30.0 | 1.15 |
| dialogue-samples | 0.80 | null | 2.0 / 30.0 | 1.15 |
| wav-raw | 0.80 | null | 2.0 / 30.0 | 1.15 |
| kuaidao | 1.20 | 0.3 | 5.0 / 28.0 | 1.50 |
| wowan | 1.20 | 0.3 | 5.0 / 28.0 | 1.50 |

**这次重构没有重新切任何一个已有项目。** 四个还在用旧参数的项目,如果
不重跑就还是原来的输出;真的重跑,会因为上面 4 条 bug 修复而和历史输出
不同(文本层面 TERMINAL 修复几乎必然触发,音频层面 edge_pad 裁剪触发率
取决于该项目组间隙有多密)。kuaidao 要不要也切换到 wowan 那套新参数、
要不要重切,还是本次会话里留的一个开放问题,重构不替你做这个决定。

## 正确性验证

没有跑一遍就信,两个真正分叉过的模块都跑了 parity test:

- **cut.py vs wowan-tts(现网、已修过 bug 的版本)**:208 个 wav **逐字节
  完全一致**,metadata.csv/filelist.txt 完全一致。
- **cut.py vs kuaidao-tts(现网、还没修 edge_pad bug 的版本)**:导出条数、
  每集分布完全一致;wav 差异全部集中在间隙<0.4s 的相邻组边界(207/374 对,
  时长差 mean 0.07s / max 0.30s,没有一条超过 1s)——和"这就是 edge_pad
  修复该有的效果"完全吻合,不是 bug。
- **silence_qa.py vs wowan-tts**:`flagged=45, head_bound=0.36,
  tail_bound=0.55`,和现网跑出来的数字完全一致。
- **punct_fix.py vs kuaidao-tts**:`replaced=117 kept_old=191`,输出文本
  逐字节完全一致。
- **prosody.py**:抽样跑 5 条,声学特征数值和本次会话里当时跑出来的
  原始记录完全一致。

跑 parity test 时顺带发现 voice-lover-tts 的 `dataset/` 和它自己当前的
`merge/groups_*.json` 对不上(用同一份重新实现的旧算法算出来的时长也和
现网 wav 对不上,说明不是这次重构引入的问题)——大概率是过去某次重新
归组之后没有重新跑 `apply_merge.py` 同步。不影响这次重构的正确性判断,
但如果之后要动 voice-lover-tts,这个不一致本身值得先弄清楚。

## 用法

```bash
# 单个阶段
python run.py stage1 kuaidao              # 全部集数
python run.py stage1 kuaidao 003 004      # 只处理指定集号
python run.py merge-prep kuaidao
python run.py merge-llm kuaidao           # 可选,需要 ANTHROPIC_API_KEY
python run.py merge-apply kuaidao
python run.py cut kuaidao
python run.py align kuaidao
python run.py punct-fix kuaidao
python run.py silence-qa kuaidao dataset          # 只出报告
python run.py silence-qa kuaidao dataset --apply  # 报告 + 剔除
python run.py loudnorm kuaidao                    # 音量归一化报告
python run.py loudnorm kuaidao --apply            # 报告 + 就地改写 wav
python run.py llm-audit kuaidao           # 可选,需要 OPENROUTER_API_KEY
python run.py prosody kuaidao

# 不含 GPU 重计算之外的全部主流程(stage1 -> ... -> punct-fix)
python run.py all wowan

# 新数据源:抄一份 projects/example.json 改路径/标签/阈值即可,不用碰代码
```

依赖需要 `voicelover` conda env(`/opt/dlami/nvme/conda-data/envs/voicelover`):
torch、silero-vad、speechbrain、qwen_asr、audio-separator、librosa、
soundfile、pyworld、requests、anthropic(merge-llm 用)。
