# RynnBrain 1.1: Towards More Capable and Generalizable Embodied Foundation Model（arXiv:2607.17977）

> 来源归档（ingest）

- **标题：** RynnBrain 1.1: Towards More Capable and Generalizable Embodied Foundation Model
- **缩写：** **RynnBrain 1.1** / **RynnBrain-VLA**
- **类型：** paper / embodied-foundation-model / vla / spatial-grounding / 3d-grounding
- **arXiv：** <https://arxiv.org/abs/2607.17977>（PDF：<https://arxiv.org/pdf/2607.17977>）
- **项目页：** <https://alibaba-damo-academy.github.io/RynnBrain>
- **代码：** <https://github.com/alibaba-damo-academy/RynnBrain>（Apache-2.0）
- **权重：** Hugging Face <https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain-11>；ModelScope <https://modelscope.cn/collections/DAMO_Academy/RynnBrain-11>
- **机构：** 阿里巴巴达摩院（DAMO Academy, Alibaba Group）；湖畔实验室（Hupan Lab）
- **骨干：** Qwen3.5 系（2B / 9B / 122B-A10B MoE）
- **入库日期：** 2026-07-21
- **一句话说明：** 在统一时空–物理 grounding 预训练下发布 **2B / 9B / 122B-A10B** 具身基础模型族；相对 1.0 新增 **接触点预测** 与（2B/9B）**原生 3D grounding**；并以 **81 维统一动作空间 + embodiment mask + flow matching** 做跨本体 **RynnBrain-VLA**，在 G1 / Astribot-S1 / Tianji-Wuji 真机长程任务上显著优于同配方 Qwen-VLA 与 π₀.₅ / GR00T N1.7。

## 摘录 1：问题与贡献

- **痛点：** 通用 MLLM 强于图文理解，但机器人需要 **3D 结构、跨视角推理、语言→物理位置 grounding**，以及能否 **迁移到真机 VLA**。
- **相对 RynnBrain 1.0：** (1) 表征更贴近操作——加 **contact-point prediction**（接触中心 + 平面抓取角）与 **native 3D grounding**；(2) 系统检验 **具身预训练初始化** 对下游 VLA 的增益。
- **贡献三条：** 统一配方下的 **2B→9B→122B-A10B** 具身 scaling；接触点 +（紧凑模型）度量 3D 框；**RynnBrain-VLA** 统一跨本体动作空间与联合多任务训练收益。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-rynnbrain-1-1.md`](../../wiki/entities/paper-rynnbrain-1-1.md)；对照 [VLA](../../wiki/methods/vla.md)、[Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)、[Foundation Policy](../../wiki/concepts/foundation-policy.md)。

## 摘录 2：架构与预训练配方

- **架构：** decoder-only VLM（Qwen3.5）；vision encoder + projector + LLM；DeepStack + Interleaved MRoPE；Dense / MoE 解码器统一输出 **文本、区域、轨迹、指点、3D、接触信号**。
- **物理输出空间：** 2D 坐标离散到 \([0,1000]\) 与语言同自回归；2B/9B 额外预测相机系 **9D 3D 框**（中心/尺寸/姿态，米与弧度，再离散化）。
- **接触点：** 从 1.0 的抓取矩形四角改为紧凑 \((p,\theta)\)——接触中心 + 平面夹爪角，避免矩形 extent 与 IoU 惩罚功能等价抓取。
- **数据混合（能力轴）：** 通用 MLLM、认知、时空定位、**3D grounding**（WildDet3D + FoundationPose）、抓取/接触、规划（含 AgiBotWorld、OXE、Galaxea 等）。
- **预训练超参（Table 1）：** AdamW；2B lr \(5\times10^{-6}\) / bs 512；9B 与 122B lr \(2\times10^{-6}\) / bs 1024。

**对 wiki 的映射：** 实体页「方法栈 / 流程总览」；与同院系 [RynnWorld-4D](../../wiki/entities/paper-rynnworld-4d-rgb-depth-flow.md) 区分「感知–推理脑 vs 4D 世界模型」。

## 摘录 3：RynnBrain-VLA 与真机评测

- **VLA 架构：** RynnBrain 骨干作 **单流 DiT**；flow matching 预测 **32-step action chunk**；指令前缀 KV cache；采用 **RTC**（每 5 步重推、\(\beta=10\)）。
- **统一 81 维动作空间：** Arm-Joint 14D / Arm-EEF 18D / Gripper 2D / Hand 40D / Torso 4D / Head 3D；**embodiment-specific mask** 只算活跃维损失。
  - **G1：** Hand 14D（双手各 7D）+ 另路 **64D SONIC** latent（不在 81D 内）→ 78D → SONIC WBC。
  - **Tianji-Wuji：** Arm-Joint 14D + Hand 40D = 54D。
  - **Astribot-S1：** Arm-Joint 14D + Gripper 2D + Head 3D + Torso 4D。
- **受控对比（同演示、同后训练 60k step）：** RynnBrain-VLA 平均 process **91.28%** / success **86.67%** vs Qwen-Based-VLA **68.33% / 60.00%**；vs GR00T N1.7 **83.31 / 73.33**、π₀.₅ **72.44 / 65.00**。
- **Generalist（联合多任务多本体）：** process **94.14%** / success **91.67%**，优于分任务微调。
- **G1 Pull the Chair：** RynnBrain-VLA **90%** vs GR00T N1.7+SONIC **75%**（同 SONIC 低层）。

**对 wiki 的映射：** [VLA](../../wiki/methods/vla.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[Qwen-VLA](../../wiki/entities/qwen-vla.md)。

## 摘录 4：Scaling 与基准要点

- **非均匀具身 scaling（相对 matched Qwen3.5）：** 一般认知双方随规模上升且差距收窄；**推理密集型认知** 上 RynnBrain 上升而 Qwen3.5 **负缩放**；**定位** 上最大 Qwen3.5 仍低于最小 RynnBrain——强调 **显式空间监督不可被纯参数缩放替代**。
- **122B-A10B：** 在 VSI-Bench / MMSI / RefSpatial-Bench 等上领先所评专有与开源模型（以论文 Table 5 为准）。
- **3D grounding：** SUN RGB-D AP@15：2B **34.28** → 9B **41.12**；WildDet3D-Bench AP3D：2B **17.36** → 9B **23.44**。

**对 wiki 的映射：** [Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)。

## 摘录 5：开源边界（项目页核查）

- **已开源：** 官方 GitHub（推理 demo、`cookbooks/` 感知/定位/接触点/3D notebook）、**2B/9B/122B-A10B** 权重（HF + ModelScope）、Apache-2.0。
- **未在公开仓见到：** RynnBrain-VLA **训练/部署代码**、VLA **权重**、真机遥操作数据集。
- **结论（截至 2026-07-21）：** **部分开源**——具身基础模型推理与权重可跑；跨本体 VLA 闭环需自建或等待后续 release。

**对 wiki 的映射：** 实体页「开源状态 / 源码运行时序图 / 局限」；[`sources/repos/rynnbrain.md`](../repos/rynnbrain.md)、[`sources/sites/rynnbrain-alibaba-damo.md`](../sites/rynnbrain-alibaba-damo.md)。

## 当前提炼状态

- [x] arXiv HTML / 项目页 / GitHub README 已对齐摘录（2607.17977）
- [x] wiki 映射：`wiki/entities/paper-rynnbrain-1-1.md` 新建
- [x] 开源边界写入 sites / repos / wiki 局限
