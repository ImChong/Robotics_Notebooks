# CausalWM: Causal Chain-of-Thought Reasoning for Embodied World Model（arXiv:2609.23184）

> 来源归档（ingest）

- **标题：** CausalWM: Causal Chain-of-Thought Reasoning for Embodied World Model
- **短名：** CausalWM（论文表格中简写 CWM；发布版本名 CausalWMv1）
- **类型：** paper（技术报告，cs.CV）
- **arXiv：** <https://arxiv.org/abs/2609.23184>（v1 提交 2026-09-19 19:16 UTC；v2 2026-09-22 07:12 UTC）
- **HTML：** <https://arxiv.org/html/2609.23184>
- **PDF：** <https://arxiv.org/pdf/2609.23184>；官方博客另链 OpenReview PDF <https://openreview.net/pdf?id=3pf4d0EEqm>（未核对其投稿会议）
- **作者：** Ziming Xu†、Shuang Liang†、Ruobing Han、Ziqiao Xi、Mingxing Rao、Kun Zhou*（通讯 / 项目负责人）、Zijun Zhang、Yuchen Yan、Yufan Wei、Junbo Huang、Yifei Shao、Fang Nan、Biwei Huang，共 13 人
- **机构：** 以太智能（Aether AI）；部分作者另署 UC San Diego、Vanderbilt University（标 ‡ 者为在 Aether AI 实习期间完成）
- **项目页：** <https://aetherlabsai.github.io/CausalWM/>
- **代码：** <https://github.com/AetherLabsAI/CausalWM>
- **权重：** <https://huggingface.co/AetherLabs-AI/CausalWM>
- **官方博客：** <https://aetherlabs.ai/articles/causalwm-causal-chain-of-thought-reasoning-for-embodied-world-model.html>（2026-09-19，见 [官方页归档](../sites/aether-causalwm.md)）
- **入库日期：** 2026-10-10
- **一句话说明：** 在 LTX-2.3-22B 视频 DiT 上做「光流 → 点图 → 未来 RGB」的显式因果思维链（stage-causal 注意力 + 逐变量去噪），31K 小时具身视频三阶段训练（预训练 / CoT 中训 / DiffusionNFT 多目标 RL）；自报 TriWorldBench 66.04 第 1（36 模型，2026-09-11 快照）、PAI-Bench-G 机器人域 89.9（本地评测，Cosmos3-Super 89.7）。

## 开源状态（步骤 2.5，2026-10-10）

| 入口 | 状态 |
|------|------|
| **代码** | **部分开源**：GitHub 仓库只含 TI2V（单图 + 文本）CoT **推理**接口（`inference.py`、`causalwm/`、改过的 `packages/ltx-core/`）；训练、RL 后训练、数据处理代码未发布 |
| **权重** | **已发布（gated）**：HF `AetherLabs-AI/CausalWM`，单文件 `CausalWMv1.safetensors`（BF16 CoT Transformer）；访问需同意条款（`gated: auto`）；创建于 2026-09-16，最后修改 2026-09-22。README 与模型卡都写「当前 checkpoint 针对 PAI-Bench-G，更多 checkpoint soon」——TriWorldBench 用的三视角动作条件模型**未发布** |
| **依赖资产** | 推理还需 Lightricks `ltx-2.3-22b-dev.safetensors`（VAE、骨干配置、文本 connector）与 Gemma-3-12B 文本编码器，各按其许可下载 |
| **许可** | LTX-2 Community License Agreement（仓库 `LICENSE`；HF `license: other / ltx-2-community`）；Gemma 部分受 Gemma Terms of Use 约束 |
| **数据** | **未公开**：31K 小时混合数据由 20 个公开数据集族整理而来，但过滤、分段、重写 caption 后的训练集与 flow / pointmap 缓存未发布 |
| **配套模型** | 同组织 HF 另有 `AetherLabs-AI/CD-LAM`（2026-09-22 创建，潜动作模型，对应 arXiv:2607.09185） |

## 核心摘录（面向 wiki 编译）

### 1. 问题与方法

- 现有具身世界模型把物理知识隐式纠缠在潜表示里，难以判断是否学到因果变量还是走了 shortcut；作者借 LLM 的 CoT，把未来预测拆成显式中间物理变量。
- 摘要原文："We introduce CausalWM, a **16B** embodied world model that performs explicit causal chain-of-thought reasoning before future video prediction."
- 结构：单个 DiT 承担全部链条；光流、点图都编码成图像，走与 RGB 相同的冻结视频 VAE；各流有独立输入 / 输出投影与模态 AdaLN；**跨流因果注意力掩码**（每个流只能读观测和前序流，后续流到前序流的路径被阻断），流内注意力双向。
- 推理逐变量去噪：先光流（运动）→ 固定后作为上下文生成 XYZ 点图（几何）→ 再生成未来 RGB。语言条件经 cross-attention，动作条件（CD-LAM 32 维潜动作）经 AdaLN。
- 中训时每次更新均匀抽一个阶段作为目标，前序变量作为干净上下文；光流用 SEA-RAFT 提取，深度与内参用 VGGT-Omega-1B-512 估计后转成归一化 XYZ 点图，离线经 VAE 编码缓存。

**对 wiki 的映射：** [paper-causalwm](../../wiki/entities/paper-causalwm.md) 方法节；[生成式世界模型](../../wiki/methods/generative-world-models.md)

### 2. 数据与训练

- 数据池：20 个来源族，采集 **31,230.8 h**，处理后保留 **19,916.2 h**（摘要写 31K，结论写 20K，分别对应两者）。最大单源 Egocentric-10K（9,980.3 h → 6,878.2 h，按光流剔除最高 30% 片段）；其余含 Ego4D、EgoDex、EPIC-Kitchens、H2O、EgoVerse、AgiBot-World Beta / 2026、Galaxea、RoboCOIN、RoboMIND、DROID、RT-1、BridgeData V2、OXE 子集、Humanoid-Everyday、RoVid-X、InternData-A1、RoboCasa365、RoboTwin 2.0。
- 处理：长度过滤 → 光流统计运动过滤 → 动作质量检查（首帧需有手 / 臂 / 夹爪）→ 事件级分段与 caption 重写（VLM / LLM 辅助）。
- Stage 1 预训练（自 LTX-2.3-22B，去掉音频部分参数）：语言条件 60k 更新、128×H100、65 帧、batch 256；动作条件变体 30k 更新、32×H200；多视角变体 120k 更新、32×H200（2–4 视角水平拼接）。
- Stage 2 Causal CoT 中训：50k 更新、64×H200、121 帧；TriWorldBench 三视角动作控制另把 URDF + 正运动学渲染的关节 / 夹爪轨迹视频作为视觉上下文控制，微调 34k 更新。
- Stage 3 多目标 RL：DiffusionNFT（类 GRPO 的组内优化），奖励含物理一致性、时间连贯、视觉质量、任务完成；报告未给出该阶段的步数与算力。

**对 wiki 的映射：** [paper-causalwm](../../wiki/entities/paper-causalwm.md) 数据与训练节

### 3. 评测（均为自报）

- **TriWorldBench**（action-conditioned、三视角、500 episode / 50 任务、19 指标）：TWB-Score **66.04**，2026-09-11 官方榜快照 36 个模型中第 1；第 2 dream4act 65.66、第 3 BWM 65.54、WoVR Plus 65.39、PhyxWM 64.26。按 19 项单指标在 36 模型中：Perspective 91.16、Image Quality 43.24 第 1，另有 5 项第 2。评测用三视角动作条件模型、自回归 129 帧窗口、30 步去噪、无 CFG。
- **PAI-Bench-G 机器人域（RO，174 prompt × 5 seed，913 道二值 VQA，Qwen3-VL-235B-A22B-Instruct 判分）**：CWM **89.9**，Cosmos3-Super 89.7（两者均为作者本地评测；CWM 使用改写后贴近预训练 caption 风格的 prompt），Veo-3 86.9、Wan2.2-I2V-A14B 81.7、Wan2.1-I2V-14B 80.1、Cosmos-Predict2.5-14B 79.9、Wan2.2-TI2V-5B 79.3、CogVideoX-5b-I2V 74.0、LTX-Video-13B 70.1（后者取自官方榜）。作者自承「与 Cosmos 3 报告一样，无法精确复现 PAI-Bench-G 榜单绝对分」。
- **少步去噪（RO）**：20/20/20 步 86.54（76.43 s）；10/10/10 86.75；8/8/8 87.53；4/4/4 88.86（24.25 s，约 3.15×）；2/2/2 88.71；1/1/1 88.84（14.81 s，约 5.16×）。
- **上下文视觉控制**：URDF 渲染的三视角控制视频替换中间流，少量微调即可让生成视频跟随给定轨迹（定性展示）。
- **案例**：bottle-to-drawer、取瓶、关抽屉三例对比 Wan2.2-A14B 与 LingBot-Video（定性）。
- 报告**没有**同骨干「无 CoT 直接预测」的消融。

**对 wiki 的映射：** [paper-causalwm](../../wiki/entities/paper-causalwm.md) 实验与评测节；[TriWorldBench](../../wiki/entities/paper-triworldbench.md)；[PAI-Bench](../../wiki/entities/paper-sa-2512-01989-pai-bench-a-comprehensive-benchmark-for-physical-ai.md)

### 4. 局限（作者自述）

- CoT 依赖预定义的中间变量集合，复杂环境下未必覆盖真实因果结构。
- 更长时程、强 OOD 物理场景的泛化有待验证。
- 目前只做世界预测，统一 world-action 模型是后续方向。

**对 wiki 的映射：** [paper-causalwm](../../wiki/entities/paper-causalwm.md) 局限与风险节
