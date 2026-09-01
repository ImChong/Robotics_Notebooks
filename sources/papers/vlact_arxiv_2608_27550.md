# VLAct（表征中心 VLA 持续预训练）

> 来源归档（ingest）

- **标题：** Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.27550>
- **项目页：** <https://starvla.github.io/VLAct/>
- **代码：** <https://github.com/starVLA/VLAct>
- **权重 / 数据：** <https://huggingface.co/StarVLA/VLAct_Qwen3_Pretrain>（集合：<https://huggingface.co/collections/StarVLA/vlact-6a903c2e0c176179da425c96>）
- **机构：** 香港中文大学（CUHK）等（Jiaya Jia、Hengshuang Zhao 等顾问；StarVLA 团队）
- **入库日期：** 2026-09-01
- **一句话说明：** 在固定机器人数据预算下，用「保留 VLM 先验 + 多头连续动作共监督 + 部分统一跨本体动作布局」做 VLA 持续预训练，16 GPU + 全开源数据即可在 LIBERO-Plus / RoboTwin / 未见本体迁移上超越大规模工业 VLA。

## 核心摘录（策展）

### 1) 问题：数据缩放之外，表征质量是瓶颈

- **摘录要点：** 机器人轨迹无法像网页图文一样无限爬取；在固定数据预算下，持续预训练必须把轨迹蒸馏为**可迁移的视觉–动作表征**，而非仅拟合动作。论文将 VLM backbone 视为 VLA 的**一阶设计变量**。
- **对 wiki 的映射：**
  - [VLAct](../../wiki/entities/paper-vlact.md) — 问题设定与总判。
  - [VLA](../../wiki/methods/vla.md) — 预训练阶段谱系。

### 2) 三条失败模式与 VLAct 配方

- **摘录要点：** 朴素 VLA 持续预训练有三类失败：（1）全参数更新侵蚀 VLM 先验；（2）单动作头导致 decoder lock-in；（3）离散 FAST 监督损失细粒度动作信息。VLAct 对策：冻结视觉编码器 + LLM 下半层 + caption 混训；OFT/PI/GR00T **多头共监督**同一 latent；**部分统一 20-D 动作布局**（共享夹爪维）+ wrap-aware 周期关节损失。
- **对 wiki 的映射：**
  - [VLAct](../../wiki/entities/paper-vlact.md) — 方法与流程图。
  - [StarVLA](../../wiki/methods/star-vla.md) — 代码基座与动作头族。

### 3) 仿真与未见本体迁移

- **摘录要点：** 对照 Qwen3VL-OFT 同骨干同下游协议：LIBERO-Plus **82.6%**（+7.6 pp）；VLA-Arena **54.8%**（+21.4）；RoboTwin 2.0 Clean **92.5%**。RoboCasa-GR1（持续预训练未见 GR-1）仅 **20%** 下游轨迹即 **49.5%**，超全数据 GR00T-N1.6 **47.6%**。RoboDojo 官方榜 success **7.60%**、score **10.66**，35 策略中第 6。
- **对 wiki 的映射：**
  - [VLAct](../../wiki/entities/paper-vlact.md) — 评测表。
  - [Robotwin](../../wiki/entities/robotwin.md) — RoboTwin 2.0 语境。

### 4) 真机 Franka 与双臂

- **摘录要点：** 桌装 Franka Research 3 + RealSense；单臂短程 in-domain **92.5%**（基线 77.5%）；长程 OOD 与双臂协调显著优于 Qwen3VL-OFT；各任务 50 演示、50k 步、8×H800 微调。
- **对 wiki 的映射：**
  - [VLAct](../../wiki/entities/paper-vlact.md) — 真机读法。

### 5) 开源状态（截至 2026-09-01，项目页核查）

- **摘录要点：** **已开源** MIT：`starVLA/VLAct` 含持续预训练脚本、下游 LIBERO/VLA-Arena/RoboTwin/DOMINO/RoboCasa 启动器、`deployment/` 真机服务；HF 发布 **VLAct_Qwen3_Pretrain** 骨干及多 benchmark 微调 checkpoint。注意：发布 checkpoint 的动作头类型与论文 headline 表未必一致（如 RoboTwin 92.5% 用 OFT，发布卡或为 GR00T）。
- **对 wiki 的映射：**
  - [vlact 仓库](../repos/vlact.md)
  - [vlact 项目页](../sites/vlact.md)

## 当前提炼状态

- [x] arXiv / 项目页 / GitHub / HF 已交叉核查
- [x] wiki 映射：`wiki/entities/paper-vlact.md`
