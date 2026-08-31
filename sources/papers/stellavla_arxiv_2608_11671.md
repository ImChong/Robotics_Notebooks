# StellaVLA: In-Context Structured Demonstration for Generalizable Vision-Language-Action Models

> 来源归档（ingest）

- **标题：** StellaVLA: In-Context Structured Demonstration for Generalizable Vision-Language-Action Models
- **短名：** StellaVLA
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.11671>
- **PDF：** <https://arxiv.org/pdf/2608.11671>
- **榜单 / 项目入口：** <https://vla-arena.github.io/#leaderboard>（论文首页链至 VLA-Arena 排行榜；**无独立 StellaVLA 项目页或官方代码仓**）
- **作者：** Siyu Xu, Yunke Wang, Zijian Wang, Dihao Zhu, Chenghao Xia, Chengbin Du, Daochang Liu, Tao Huang, Chang Xu
- **入库日期：** 2026-08-31
- **一句话说明：** 离线 VLM 把原始示范转成「任务计划 + 子目标 + 2D/3D 运动 verbalization」的结构化 in-context 示范；并行双专家训练、推理仅 action expert，单次检索示范即可 OOD 适应。

## 开源状态（步骤 2.5）

- **确认未开源（StellaVLA 本体）**：截至入库日 arXiv 与 VLA-Arena 页 **未列出 StellaVLA 官方 GitHub / 权重**；论文实现基于 **Qwen3-VL-4B + OpenVLA-OFT 风格 action expert**，对照基线 **StarVLA-OFT** 见 [StarVLA（arXiv:2604.05014）](https://arxiv.org/abs/2604.05014) 社区代码库，**非 StellaVLA 官方发布**。
- **相关开源资源**：**VLA-Arena** 基准框架与数据集公开于 <https://vla-arena.github.io>（PKU 等）。

## 核心摘录（面向 wiki 编译）

### 摘录 1：结构化 in-context 示范 vs 原始轨迹模仿

- **问题：** VLA 在场景 / 视角 / 物体 OOD 时性能崩塌；传统 ICIL 只堆叠原始 \((o,a)\) 轨迹，策略模仿「做了什么」而非「为什么」，易退回预训练先验（behavioral inertia）。
- **关键想法：** 自动离线管线把每条原始轨迹 \(\tau=\{(o_t,a_t)\}\) 转成 **结构化示范**：高层 **任务计划**、段级 **子目标语义**、细粒度 **2D/3D 运动 verbalization**（零人工标注；用 Qwen3-VL 因果推断分段）。
- **跨具身：** 真机、人手、XR 重定向示范统一成同一结构化表示；异具身示范 **仅作 context**，可执行动作仍在目标机器人原生控制空间监督。

**对 wiki 的映射：** [paper-stellavla-structured-icl-vla](../../wiki/entities/paper-stellavla-structured-icl-vla.md)、[机器人 In-Context Learning](../../wiki/concepts/robot-in-context-learning.md)、[VLA](../../wiki/methods/vla.md)

### 摘录 2：并行双专家训练与推理去语言解码

- **骨干：** Qwen3-VL-4B-Instruct + OpenVLA-OFT 风格 **MLP action expert**（\(L_1\) chunk 回归）。
- **并行 spatial-language expert：** 交叉熵（\(\lambda=0.3\)）监督当前 **子任务** 与同一 action chunk 的 **2D/3D 运动描述**；与 action expert **共享 backbone 表征**。
- **推理：** **移除 spatial-language expert**；结构化示范 prefix **KV-cache** 一次编码；控制环 **无自回归语言解码**，真机 pipeline 约 **205 ms/chunk**（模型侧 88–91 ms）。
- **对照：** 同骨干同数据的 **StarVLA-OFT**（无检索示范、无 spatial-language 监督）衡量完整设计增益。

**对 wiki 的映射：** [BPP](../../wiki/entities/paper-behavior-prompting-policy.md)（示范作 prompt）、[RoboTTT](../../wiki/entities/paper-robottt-test-time-training-vla-context.md)（TTT 写 fast weights，机制不同）

### 摘录 3：仿真与真机评测

- **LIBERO：** 平均 **98.8%** SR（Spatial 99.6 / Object 99.0 / Goal 99.6 / Long 96.8）；相对 StarVLA-OFT Goal **+3.4**、Long **+3.0**。
- **VLA-Arena（2026-08-01 榜）：** Overall **0.63**（\(\pi_{0.5}\) **0.44**、LingBot-VLA **0.22**）；L0/L1/L2 均值 **0.84 / 0.62 / 0.43**。
- **LIBERO-Plus（零样本）：** 平均 **85.1%**（+10.1 vs StarVLA-OFT）；视角 / 噪声 / 机器人初态增益最大。
- **真机：** 可用机器人示范、人手示范、XR 人→机示范作 structured context，适应 OOD 任务（详见论文 §4.3）。

**对 wiki 的映射：** [manipulation](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-stellavla-structured-icl-vla.md`](../../wiki/entities/paper-stellavla-structured-icl-vla.md)
- 与 [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md)、[RoboTTT](../../wiki/entities/paper-robottt-test-time-training-vla-context.md)、[Zero-WAM](../../wiki/entities/paper-zero-wam.md) 构成「部署期适应」对照轴（结构化 ICL vs TTT fast weights vs 人视频 WAM 规格）

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与 ICL 概念回链
