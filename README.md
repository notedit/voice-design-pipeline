# voice-design-pipeline

通用的 TTS 训练数据处理流水线:从原始长音频出发,完成 BGM 分离、VAD 分段、
说话人聚类、ASR 转写、语义归组、重切导出、强制对齐、标点校正、静音 QA、
音量归一化和可选的 LLM 质量审计,产出可直接用于 TTS 训练的 dataset。

每个数据源只需要一份 json 配置(路径、说话人标签、各阶段阈值),共享同一套
代码;不建议通过复制项目目录的方式接入新数据源。

## 目录结构

```
voice-design-pipeline/
  ttspipe/
    config.py       ProjectConfig:每个数据源一份 json,见 projects/
    gpu.py           挑空闲显存最多的 GPU
    stage1.py        分离 BGM -> VAD 分段 -> ECAPA 聚类锁定主说话人 -> ASR
    merge_prep.py    report -> 归组输入 + 贪心归组基线
    merge_llm.py     [可选] LLM 语义归组(API)
    merge_apply.py   应用语义拆分(如果有)+ 校验
    cut.py           按 groups 重切音频,压缩组内静音,产出 dataset/
    align.py         全量强制对齐 -> alignments.jsonl
    punct_fix.py     按实测停顿校正标点
    tail_punct.py    尾标点如实化:被时长上限强制断开的条目,结尾"。"改","
    para_cut.py      剪除对齐 gap 内未转写声音(笑声/语气声),响度+对齐驱动
    silence_qa.py    静音异常检测(IQR + 固定阈值),可选剔除
    denoise.py       [可选] 语音增强降噪(MossFormer2_SE_48K),带 ECAPA
                     音色相似度验收,低于阈值保留原音频
    loudnorm.py      音量归一化(项目内集级增益),可选就地改写
    llm_audit.py     [可选] LLM 打分+判定
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
(`distance_threshold=0.6`)锁定目标说话人的簇,其余全部丢弃
-> ASR 批量转写,转写失败(<2 字或不含中文)的段标记丢弃。
这些参数在实践中很少需要调整,配置里仍然放出来(`ProjectConfig.stage1`)
是为了让新项目在真的需要不同灵敏度时不用改代码,不代表鼓励每个项目都调。

目标说话人簇的两种选法:
- 默认(`stage1.ref_audio=null`):选总时长最大的簇——适合"主播就是目标"
  的单人主导型音源。
- 参考模式(`ref_audio` 指向目标说话人几秒~几十秒的干净样本):对参考算
  ECAPA 向量,选与它余弦距离最近的簇;最近距离仍超过 `ref_max_dist`
  (默认 0.60)时判定该集没有目标说话人,整集不保留任何段(report 里
  `target_found=false`),宁缺毋滥。目标说话人不是说话最多的人、或不同
  集的主次说话人会换人时,必须用参考模式。
- 一集聚出多个簇时,自动往 `work/spk_preview/<集号>/cluster_<k>.wav` 导出
  各簇 ~12s 的试听拼接;没有参考样本时可先跑一遍,人工听预览指认目标,
  再把选中的预览 wav 填进 `ref_audio` 重跑。report 的 `spk_clusters` 字段
  记录每簇的段数/总时长/与参考的距离,选择过程可审计。

**归组(merge_prep + merge_llm + merge_apply)**
贪心基线的产品策略(默认):**单条硬上限 30s,优先区间 5-20s**。相邻段
间隙 ≤2.5s 就可合并;组打包到 20s 后进入 20~29s 弹性区,只在间隙 ≥0.5s
的"好断点"(句末/分句停顿)断开;到 29s 硬上限(给 pad 留 ~1s 余量,导出
后 ≤30s)仍没有好断点,回溯到组内最大间隙处断。配套地,stage1 切超长段
用分级候选断点(≥0.5s -> ≥0.3s -> ≥0.15s 保底)——两者共同把断点尽量推
向句末,降低"切在逗号上"的比例(实测语速密的讲课源仍有 ~2/3 边界是强制
断开,tail_punct 会把这些条目的尾"。"如实化为",")。
`pref_max_dur=null` / `split_tiers=null` 关闭,退回旧的"直接打包到上限"
行为,仅用于历史项目 parity 复现。LLM 语义归组是贪心基线之上的可选复审
——不跑的话贪心基线直接生效,流水线不会卡住。`validate_groups` 保证最终
分组覆盖完整、合并条件满足、时长不超限。

**cut(按组重切音频,`ttspipe/cut.py`)**
- 组内静音压缩:间隙超过 `max_gap` 才压缩。`gap_pass_ratio=None` 时压平到
  固定 `max_gap`;给一个 0~1 的比例时,超出部分按比例保留差异
  (`target = max_gap + (gap-max_gap)*ratio`),原始间隙越长压缩后还是越长,
  只是打折——不让所有长停顿在最终数据里长度一样。拍平压缩会把有真实长短
  变化的自然停顿全部拍进同一个窄区间,TTS 学出的句间停顿会千篇一律。
- 压缩边界用响度回退(`energy_trim_end/start`),不是照搬 VAD 声明的边界——
  VAD 判"有没有人声活动"不是判响度,句尾自然渐弱时 VAD 边界常比响度真正
  跌破底噪晚 0.3-0.6s,不回退会导致导出音频里出现异常长的连续静音。
- 相邻两组的首尾 `edge_pad` 会互相裁到真实边界中点,不会越界到对方的
  真实内容里。快节奏对话里两句话之间常常只隔 0.1~0.3s,固定 pad 0.2s
  会把下一组开头的话录进这一条尾巴;长句被时长上限强制切断的场景,受
  影响的边界对比例也会很高。
- 边缘淡入淡出(`fade_in`/`fade_out`,产品默认各 60ms):切片头部从零
  半余弦淡入、尾部淡出平滑到静音——源有底噪时,边界硬切会产生可听的
  咔哒/底噪突起。para-cut 剪过头/尾的条目会重新套用同样的淡化。
  0 = 关,仅用于历史项目 parity 复现。
- 句尾静音保底(`min_tail_sil`):每条导出音频按响度实测真实语音结束点,
  句尾静音不足 `min_tail_sil` 秒时补零(带 20ms 淡出)到该值。补零而不是
  往源音频里扩,因为上一条的中点裁剪正是为了不吃到邻组内容——间隙
  <2*edge_pad 时源音频里本来就没有足够的静音可取。实测多数数据集里句尾
  静音 <300ms 的条目能占到两成到五成以上,不是个别现象。测量帧从尾部
  对齐(残余采样落在头部),从头部对齐会把"说话持续到文件末尾"的最后
  不足一帧误当静音,补零量算少。None = 关闭。

**silence_qa(静音异常检测,`ttspipe/silence_qa.py`)**
- 句首/句尾静音用 IQR(Q3+1.5*IQR)定异常边界,数据驱动,每个项目按自己
  的分布重新算。
- 句中/句末停顿(按夹在中间的标点分类)用固定阈值 `fixed_gap_th`——这组
  绝大多数值接近 0,IQR 在这种分布上不适用。这个阈值必须跟 cut 阶段的
  `max_gap`/`gap_pass_ratio` 配套:压缩放宽后合理停顿会变长,阈值要一起
  放宽,两边不能只改一边。
- 文本-时间戳匹配是 token 级指针匹配,不是逐字符——强制对齐对英文/数字串
  (比如"SUV")会整体输出成一个多字符 item,逐字符比较会让这类 item 永远
  匹配不上,把它前面的字误判成"最后一个对齐字",拼出虚假的句尾静音。

**punct_fix(标点校正)——已知限制**
文本匹配是逐字符顺序指针 walk,没有重新同步能力:新旧文本在靠前的位置
一旦有一处标点/字符对不上,后面全部错位,覆盖率跌破 80% 阈值就整条退回
旧文本、不做任何校正。长句(20s+)+ 口语化内容(重复、自我修正)命中率
明显更高。需要的话应该单独立项,换成真正的 diff/对齐算法。

**loudnorm(音量归一化,`ttspipe/loudnorm.py`)**
实测(active-speech RMS 口径)发现响度不一致的大头通常在"集与集之间",
不在集内:集中位响度可以横跨 15-20 dB,而集内 p5-p95 只有 5-10 dB。
集间差异是录音/上传增益不一致,放心拉平;集内波动含说话人真实强弱,
逐条拍平会重蹈拍平压缩停顿的教训。所以:
- 默认只做集级增益,目标 = 本项目各集响度中位数(各项目分开训,不需要
  跨项目统一水平;真要指定用 `loudnorm.target_db`)。集内偏差想收敛用
  `utt_pass_ratio`(注意语义和 gap_pass_ratio 方向相反:这里 None=完全
  保留,0=拍平)。
- 做成独立后置阶段跑在 dataset/wavs 上,不塞进 stage1/cut 前面——cut 的
  `energy_floor_db` 和 VAD 读的是绝对 dB,前置归一化会改变它们的输入;
  后置还能直接处理已有数据集,不用重切。
- 增益被峰值余量钳制(`peak_ceiling_db`,默认 -1 dBFS),永不削波;到不了
  目标的集在报告里逐集写明 n_clamped/achieved_db,不做静默截断。
- `--apply` 就地改写 wav,逐条实际增益记录在报告 applied_gains 里,这份
  记录是撤销的依据。已知限制:utt_pass_ratio 生效时二次 apply 不幂等
  (偏差再乘一次 ratio),集级模式天然幂等。

**llm_audit / prosody(可选的两条尾巴)**
`llm_audit.py` 的 schema:韵律分+自然度分+四项排除标准(多说话人/爆音/
异常停顿/口齿不清),人物描述从 `ProjectConfig.llm_audit_speaker_desc`
读——单人讲课、两人对话、有声书朗读等场景差异很大,写死在 prompt 里会
答非所问。`prosody.py` 是纯声学的韵律变化指标(F0/能量/停顿的 z-score),
和 `llm_audit` 的主观打分是两套不同的东西,字段名分别是
`acoustic_prosody_score` 和 `llm_prosody_score`,不要合并成一个。

## 实现细节里容易踩的坑(已写死修复,不放进配置)

1. **TERMINAL 标点集须同时收全角+半角**(`cut.py`)。只收半角标点会让
   `join_texts()` 对全角标点结尾的中文文本误判"没有标点结尾",叠加插入
   多余逗号(产生"，,"这种双标点)。
2. **重跑前清空 `WAV_OUT`**(`cut.py`)。归组变化会让 seg_id 编号挪位,
   不清空会在目录里留下不属于当前 metadata.csv 的孤立 wav。
3. **`edge_pad` 邻组裁剪**(`cut.py`),见上面 cut 阶段的说明。
4. **token 级指针匹配**(`silence_qa.py`),见上面 silence_qa 阶段的说明。
5. **重切时删除过期的 alignments.jsonl**(`cut.py`)。align 的断点续跑按
   seg_id 跳过已有条目,重切后同名 id 对应的是新音频,旧对齐若保留会被
   punct_fix/silence_qa 拿去用,产出错误的文本替换与静音判定。

## 配置里保留的旋钮(逐项目取舍,不是 bug)

不同类型的数据源适合不同参数,示例:

| 场景 | max_gap | gap_pass_ratio | min/max_dur | fixed_gap_th |
|---|---|---|---|---|
| 朗读/有声书类 | 0.80 | null(拍平) | 2.0 / 30.0 | 1.15 |
| 讲课/对话类(保留停顿层次) | 1.20 | 0.3 | 5.0 / 28.0 | 1.50 |

## 用法

```bash
# 单个阶段(<project> 对应 projects/<project>.json)
python run.py stage1 myproject              # 全部集数
python run.py stage1 myproject 003 004      # 只处理指定集号
python run.py merge-prep myproject
python run.py merge-llm myproject           # 可选,需要 ANTHROPIC_API_KEY
python run.py merge-apply myproject
python run.py cut myproject
python run.py align myproject
python run.py punct-fix myproject
python run.py silence-qa myproject dataset          # 只出报告
python run.py silence-qa myproject dataset --apply  # 报告 + 剔除
python run.py loudnorm myproject                    # 音量归一化报告
python run.py loudnorm myproject --apply            # 报告 + 就地改写 wav
python run.py llm-audit myproject           # 可选,需要 OPENROUTER_API_KEY
python run.py prosody myproject

# 主流程一把跑完(stage1 -> ... -> punct-fix)
python run.py all myproject

# 新数据源:抄一份 projects/example.json 改路径/标签/阈值即可,不用碰代码
```

## 依赖

Python 环境需要:torch、silero-vad、speechbrain、audio-separator、librosa、
soundfile、pyworld、requests、anthropic(merge-llm 用),以及所选的 ASR 库。
