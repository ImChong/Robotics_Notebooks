# PACE: Persona Adaptation through Conversational Elicitation in Human-Robot Interaction（通过对话引导实现人机交互中的人格适配）

> 一手论文原文归档；本文件为结构化核读笔记，不转载论文全文。

- **论文：** https://arxiv.org/abs/2607.15579
- **原文 HTML（核读版本）：** https://arxiv.org/html/2607.15579 （v5，2026-10-08）
- **PDF：** https://arxiv.org/pdf/2607.15579
- **项目主页：** https://lipzh5.github.io/PACE/ （归档：[pace-persona-adaptation.md](../sites/pace-persona-adaptation.md)）
- **GitHub：** https://github.com/lipzh5/PACE （归档：[lipzh5-pace.md](../repos/lipzh5-pace.md)）
- **作者：** Peizhen Li、Longbing Cao、Megani Rajendran、Timothy Liu、Aik Beng Ng、Simon See
- **机构：** Macquarie University（麦考瑞大学）；NVIDIA Singapore（英伟达新加坡）
- **发表：** arXiv 预印本，2026-07-17 首投、2026-10-08 v5；作者 README 和 arXiv 标注已被 Humanoids 2026 接收。
- **入库：** 2026-11-16
- **核对范围：** 逐节核读 arXiv v5 HTML：Abstract、`I 引言、`II 相关工作、`III-A/B/C 方法、`IV-A/B/C/D 实验、`V 结论和局限；核对 Fig.1–5、Table I–III、参考文献列表。未核实实际机器人可执行源码与复现实测。
- **通读：** 已完成

## 问题与方法（`I–III）

- 问题：静态开发者 system prompt 难以随不同用户及情境更换社交人格；PACE 的适配对象是**交互人格与表达风格**，并不是运动策略或 VLA 动作控制。
- **交互问答**：以 5 个开放式锚点问题为基础，LLM 根据回答的信息密度和心理线索生成分支追问；文中平均每位参与者追加约 1 个自适应问题，整个 elicitation 平均时长约 4.5 分钟。
- **结构化人格**：麦克风收音，Google Cloud Speech-to-Text 转录。LLM 从 Traits（HEXACO）、Values（Schwartz）、Motivation（SDT / McClelland）、Orientation（RFT）、Identity 与 Policies（CAPS IF/THEN）提炼 PersonaSpec JSON；多视角分析以社交心理学、行为经济学及社会学为提示视角，并不等于真实专家分别核验。
- **人格激活**：GPT-5.5-mini 生成 PersonaSpec；编译为系统提示词，更新 Ameca 的 LLM agent state。Amazon Polly 合成语音；GPT 类情绪推断选择 Ameca 预定义的 7 类面部情绪动画，并与语音 viseme（口型）混合。
- **音频运行约束**：OpenAI 流式 API 降低体感延迟；独立 ASR 进程可暂停、恢复，避免扬声器播报时机器人误识别自己的声音；时间戳用于消解重叠输入。
- **实体边界**：论文明确重点是**口语和面部表情同步**，非全身动作合成；不应将“persona switch”误称“真机运动控制策略迁移”。

## 实验与统计（`IV）

- **被试与设计**：N=25（12 男、13 女，21–54 岁），实验室内与 Ameca 面对面；4.5 分钟访谈后，10 分钟非结构化协作规划任务；同一被试比较 Static Baseline 与动态 PACE，顺序平衡。
- **人格保真**：参与者预先填写问卷/行为任务，两周内重复形成对照；隔离机器人硬件后，让生成的人格完成 GSS、BFI-44、经济博弈及社交情境文本任务。
- **Table II 关键数值（静态 → PACE）：**
  - BFI-44 五维 Pearson r：0.389 → **0.939**；MAE：1.220 → **0.128**，均 p<0.001。
  - 社交情境 Top-1：73.33% → **94.67%**（p=0.001）。
  - 经济博弈金钱决策 MAE：1.800 → **0.440**（p<0.001）。
  - GSS 态度 Top-1：80.00% → 88.00%（p=0.097，**未达到** 0.05 显著水平）。
  - 经济博弈二元选择匹配：88.00% → 84.00%（p=0.317，**未改善**）。
- **Table III 用户感知 HRI（5 分 Likert，静态 → PACE）：** 信任 3.52→4.18（p=0.012）；拟人化 3.28→4.05（p=0.008）；人格一致性 3.35→4.32（p<0.001）；个人相关性 3.08→4.41（p<0.001）；总体质量 3.46→4.26（p=0.004）。
- 显著性来自论文报告的配对 t 检验 / Wilcoxon 检验；这是作者研究数据，不是独立复现。

## 局限（`V）

1. 云端识别和推理可能有噪声、转写误差和延迟。
2. 几分钟的初始访谈不能覆盖数周、数月的偏好变化。
3. 表情受 Ameca 的预定义表情库与机构限制；未证实跨机器人迁移或生成式全身动作。
4. N=25、控制实验室和主观 Likert 量表限制外部有效性；没有提供真实业务长期部署指标。
5. 官方仓库 README 列出的运行命令对应脚本当前不在仓库中，无法据此做源码级复现。

## 对知识库的映射

- 主节点：[paper-pace-persona-adaptation.md](../../wiki/entities/paper-pace-persona-adaptation.md)
- 交叉对照：[人形机器人智能语音交互](../../wiki/methods/humanoid-voice-interaction.md)（PACE 在语音闭环上增加人格个性化与面部表达）
