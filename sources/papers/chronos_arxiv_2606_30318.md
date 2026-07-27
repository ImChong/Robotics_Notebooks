# Chronos: A Physics-Informed Full-History Framework for Non-Markovian Long-Horizon Manipulation

> 来源归档（ingest）

- **标题：** Chronos: A Physics-Informed Full-History Framework for Non-Markovian Long-Horizon Manipulation
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页与 GitHub 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2606.30318>
  - <https://ar5iv.labs.arxiv.org/html/2606.30318>
  - <https://chronos-manipulation.github.io/>
  - <https://github.com/yulinzhouZYL/Chronos>
  - <https://huggingface.co/yulinzhouZYL/Chronos-RMBench>
- **作者：** Yulin Zhou, Yimeng Wang, Nengyu Wang, Shaojia Xing, Shiyun Tu, Xiang Li, Jingkai Zhang, Ningbo Jiang, Yuankai Lin, Hua Yang, Xiangrui Zeng, Zhouping Yin
- **机构：** 华中科技大学（HUST）机械科学与工程学院；通讯 Hua Yang
- **venue / 状态：** 已投 IEEE Transactions on Robotics (T-RO)（仓库 README）
- **入库日期：** 2026-07-27
- **一句话说明：** 把观测历史升格为策略动力学的 **潜状态**：每物理控制步一个状态 token，用 **选择性 SSM（Mamba 式）** 全历史因果传播，经 **IMLE** 生成多模态粗动作先验，再以 **二阶 Schrödinger 启发加速度桥** 精炼；RMBench 平均 **73.6%**（相对 π₀.₅ **+62.4 pt**、相对 Mem-0 **+22.8 pt**，参数约 **0.3B**），真机双臂四任务平均 **78%**。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 项目页 | <https://chronos-manipulation.github.io/> — Overview / Framework / Results / 仿真与真机视频 / BibTeX；链到 arXiv 与代码 |
| GitHub | <https://github.com/yulinzhouZYL/Chronos> — MIT；含 `RMBench/policy/Chronos` 与 `real_wolrd/` 双臂 UR3 采数/训练/闭环 |
| 权重 | Hugging Face **Chronos-RMBench** 已释出多任务 `last.ckpt` + scaler（EE 16D） |
| Coming soon | 清理后的 **ALOHA**、**RoboTwin 2.0** 基准代码（README） |
| 结论 | **已开源（部分）** — RMBench 仿真 + 真机 UR3 管线与 RMBench ckpt 可用；ALOHA / RoboTwin2.0 清理版待发 |

## 核心论文摘录（MVP）

### 1) 问题：观测别名下的非马尔可夫模仿

- **链接：** <https://arxiv.org/abs/2606.30318> §I–III
- **摘录要点：** 许多 VLA / 生成式模仿仍条件于当前帧或短窗；在「取杯—检查—复位」等任务上，复位前后画面可几乎相同，但正确动作不同——**观测空间非马尔可夫**。正确目标是 $a_t^\star=\pi^\star(\mathcal{H}_t)$，而非仅 $\pi(o_t,p_t)$。
- **对 wiki 的映射：**
  - [Chronos（论文实体）](../../wiki/entities/paper-chronos.md) — 问题定位。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 长程 / 记忆依赖操作。
  - [VLA](../../wiki/methods/vla.md) — 与 Markovian VLA、显式记忆 VLA 对照。

### 2) 方法：全历史状态 token + IMLE 先验 + 二阶加速度桥

- **链接：** arXiv §III–IV；项目页 Framework
- **摘录要点：**
  - **状态 token：** $\bm{x}_t=\phi_\eta(o_t,p_t)$，序列长度等于轨迹物理长度 $L$（非 patch 数）。
  - **SSM：** $(\bm{h}_{1:L},\bm{y}_{1:L})=\mathrm{SSM}_\theta(\bm{X}_{1:L})$；训练保留完整时间反向传播（相对 MTIL 的 detached 状态）。
  - **IMLE：** 历史条件隐式生成器采样粗多模态动作 chunk 先验 $\bm{q}_0$。
  - **二阶桥：** Madelung 变换 + Kostin 耗散启发；预测加速度场，立方参考路径 + **四次 bell 噪声日程**（端点位置与速度扰动为零）；推理用 symplectic Euler 积分数步（ALOHA 上 3–5 步最优）。
  - **目标：** $\mathcal{L}=\mathcal{L}_{\mathrm{IMLE}}+\lambda_{\mathrm{acc}}\mathcal{L}_{\mathrm{acc}}+\lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{BC}}$。
- **对 wiki 的映射：**
  - [Chronos](../../wiki/entities/paper-chronos.md) — 流程总览与工程表。
  - [Action Chunking](../../wiki/methods/action-chunking.md) — chunk 作广义坐标。
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md) — 一阶 score/velocity 对照。

### 3) 仿真：ALOHA / RoboTwin 2.0 / RMBench

- **链接：** arXiv §V；Table II–V
- **摘录要点：**
  - **ALOHA 双臂插入（50 demo / 50 trial）：** Chronos **90%**；同历史编码器下 diffusion / flow 头 **66% / 72%**；去掉 SB **86%**；一阶噪声日程 **72%**。
  - **RoboTwin 2.0 Easy（8 任务）：** Chronos 平均 **70.0%**（DP3 **59.5%**，π₀ **48.8%**）；作者强调此基准偏 **当前几何充分可观测**。
  - **RMBench（7 记忆任务）：** Chronos **73.6%** vs π₀.₅ **11.2%**（**+62.4 pt**）、Mem-0 **50.8%**（**+22.8 pt**）；Press Button / Battery Try 仍极难（OCR/接触脆性）。
- **对 wiki 的映射：**
  - [Chronos](../../wiki/entities/paper-chronos.md) — 评测表。
  - [RoboTwin](../../wiki/entities/robotwin.md) — RoboTwin 2.0 / RMBench 语境。
  - [EventVLA](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) / [KEMO](../../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) — 稀疏视觉记忆对照路线。

### 4) 真机：双臂 UR3 + 单 RGB

- **链接：** arXiv §V-F；Table VI；项目页 Real-World
- **摘录要点：** 双 UR3 + D435 单目；ResNet18 冻结骨干（不用点云，因小盖/接触处深度噪）。四任务各 50 trial：平均 **78%**（π₀.₅ **7%**）；三项记忆依赖平均 **72%**（π₀.₅ **0%**）。Swap-T vs Swap-T-Mem 对照隔离观测别名；Cover Blocks 真机仅 **20%**（归因感知接口而非记忆机制失效）。
- **对 wiki 的映射：**
  - [Chronos](../../wiki/entities/paper-chronos.md) — 真机表与局限。
  - [Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md) — 双臂协调语境。

### 5) 局限与开放边界

- **链接：** arXiv §V-G、§VI；README Coming Soon
- **摘录要点：** Press Button 缺 OCR；Battery Try 接触精度主导失败；真机 Cover Blocks 需更强 2D 骨干或更高质量深度；ALOHA / RoboTwin2.0 清理代码尚未发布。作者主张：下一阶段策略应同时缩放 **时间结构与物理结构**，而非仅参数与数据。
- **对 wiki 的映射：**
  - [Chronos](../../wiki/entities/paper-chronos.md) — 局限与开源边界。
