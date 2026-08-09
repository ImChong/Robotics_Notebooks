# Immersive Social Interaction with VR and LLM-Assisted Humanoids（arXiv:2607.07430）

> 来源归档（ingest）

- **标题：** Immersive Social Interaction with VR and LLM-Assisted Humanoids
- **类型：** paper / humanoid / teleoperation / VR / LLM / social-interaction / multimodal-data
- **arXiv abs：** <https://arxiv.org/abs/2607.07430>
- **PDF：** <https://arxiv.org/pdf/2607.07430>
- **HTML：** <https://arxiv.org/html/2607.07430>
- **项目页：** 无独立 `*.github.io` / 机构 lab 项目页（截至 2026-08-09）
- **代码：** **确认未开源** — 论文未列官方训练/部署仓；组件引用第三方 [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)、[LiveKit Agents](https://github.com/livekit/agents)、[Silero](https://github.com/snakers4/silero-models) 等，不可当作本文系统复现入口
- **机构：** 纽约大学阿布扎比分校（NYU Abu Dhabi）
- **作者：** Niraj Pudasaini、Geeta Chandra Raju Bethala、Pranav Doma、Anthony Tzes、Yi Fang
- **发表 / 上传：** 2026-07-08（arXiv v1）；注释为 IEEE-RAS Humanoids Workshop: Designing Interactive Humanoids
- **硬件：** Apple Vision Pro；Unitree H1 + Inspire Robotics 灵巧手；ROS 1 双向音频
- **入库日期：** 2026-08-09
- **最后更新：** 2026-08-09
- **一句话说明：** 用 **Apple Vision Pro + LLM 语音高层 locomotion + VR 腕/指跟踪操作 + 双向音频社交** 做全身人形遥操作，并同步录多模态数据供后续模仿学习；H1 上新手物体操作 **80%**、社交传方块 **70%**。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2607.07430>
- **核心贡献：** 既有人形遥操作常要求全身动捕（体力负担）或低层多关节控制（认知负担）。本文给出沉浸式接口：
  1. **语音 → 高层 locomotion**（LLM 解析为 `move/rotate/stop/stand`，底层用预训练双足 RL）；
  2. **VR 腕/指跟踪 → 臂与灵巧手**（坐标系变换 + Pinocchio IK + PD）；
  3. **双向音频**支持远程社交/telepresence；
  4. 同步录 egocentric RGB、语音/文本、关节、手势、眼动等，服务下游模仿学习。
- **对 wiki 的映射：**
  - [论文实体](../../wiki/entities/paper-immersive-social-vr-llm-humanoids.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 2) 语音 locomotion 栈（§II-A）

- **链接：** arXiv HTML §II-A
- **核心贡献：** Vision Pro 流 egocentric **640×480**；Deepgram STT → **GPT-4** 解析高层命令 → Silero TTS + LiveKit Agents；不确定时二次确认。可选 GPT-4V 场景描述默认关闭以保延迟。双足策略引用 ExBody / RMA 类预训练 locomotion。
- **对 wiki 的映射：**
  - [论文实体](../../wiki/entities/paper-immersive-social-vr-llm-humanoids.md) — 流程总览
  - [全身控制](../../wiki/concepts/whole-body-control.md)

### 3) 操作与社交（§II-B / §II-C）+ 评测（§III–IV）

- **链接：** Methodology / Results
- **核心贡献：**
  - 操作：VisionProTeleop 流 SE(3) 腕/指 → 每臂 **4 DoF** + 每手 **6 DoF**；Pinocchio IK → PD。
  - 社交：ROS 1 双向音频；任务为口头要方块→步行递交→握手。
  - Table I：物体抓放新手/专家 SR **0.8 / 0.90**（时间 52 / 22 s）；社交传方块 **0.7 / 0.8**（326 / 158 s）。
  - Table II：相对 Open-TeleVision / HumanPlus / Human-to-Humanoid，本文强调同时覆盖 **语音 locomotion + 操作 + 社交交互**。
- **对 wiki 的映射：**
  - [Open-TeleVision](../../wiki/entities/paper-loco-manip-161-131-open-television.md)
  - [HumanPlus](../../wiki/entities/paper-loco-manip-161-012-humanplus.md)
  - [H2O / human-to-humanoid](../../wiki/entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md)
  - [模仿学习](../../wiki/methods/imitation-learning.md)

### 4) 开源与复现边界（步骤 2.5）

- **链接：** 全文 + arXiv 元数据（无项目页）
- **核心贡献：** **确认未开源**。无官方 GitHub / 权重 / 数据集发布；依赖第三方 Vision Pro 遥操作与语音栈。复现需自组 H1 + Inspire 手 + AVP + ROS 音频与 LLM API。同组相关工作含 H2-COMPACT（Humanoids 2025）与 embodied chain-of-action 等，但非本系统代码。
- **对 wiki 的映射：**
  - [论文实体 · 局限与工程实践](../../wiki/entities/paper-immersive-social-vr-llm-humanoids.md)
  - [H2-COMPACT](../../wiki/entities/paper-loco-manip-161-062-h2-compact.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-immersive-social-vr-llm-humanoids.md`](../../wiki/entities/paper-immersive-social-vr-llm-humanoids.md)
- 互链参考：[Teleoperation](../../wiki/tasks/teleoperation.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Open-TeleVision](../../wiki/entities/paper-loco-manip-161-131-open-television.md)、[HumanPlus](../../wiki/entities/paper-loco-manip-161-012-humanplus.md)、[H2O](../../wiki/entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md)、[Teleopit](../../wiki/entities/paper-teleopit.md)、[模仿学习](../../wiki/methods/imitation-learning.md)、[宇树](../../wiki/entities/unitree.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
- [x] 开源状态核查（无项目页 / 无官方仓）
