---
type: entity
title: "HITCH"
project_id: hitch-vlm-humanoid-harness
arxiv: "2611.08039"
tags: [paper, humanoid, vlm, whole-body, loco-manipulation, agent, tsinghua]
status: complete
updated: 2026-11-09
project: https://hitch.axell.top
related:
  - ../methods/vla.md
  - ../concepts/whole-body-control.md
  - ../tasks/loco-manipulation.md
  - ./paper-heft.md
  - ./nvidia-sonic.md
  - ./paper-loco-manip-161-075-simple.md
  - ./paper-loco-manip-161-135-amo.md
  - ./paper-loco-manip-161-156-psi0.md
  - ./paper-voxposer.md
  - ./qwen3-vl.md
  - ./unitree-g1.md
sources:
  - ../../sources/papers/hitch_arxiv_2611_08039.md
original:
  - ../../sources/papers/hitch_arxiv_2611_08039.md
summary: "HITCH（arXiv:2611.08039）：给 VLM 一套感知（选图像点取 3D 位置）加执行（按身体部位下命令，规划器加学习跟踪器变全身动作）的 harness，用仿真教师在 5 个基础任务上的轨迹微调，无目标任务轨迹时 SIMPLE 51.1%、HumanoidArena 子集 42.5%，真机 G1 搬箱 13/20。"
---

# HITCH（arXiv:2611.08039）

**HITCH**（*HITCH: Teaching a Vision-Language Model to Operate a Humanoid through a Whole-Body Harness*，[arXiv:2611.08039](https://arxiv.org/abs/2611.08039)，清华大学、中国科学院自动化研究所、上海期智研究院，2026-11-05）：全文通读归档见 [sources/papers/hitch_arxiv_2611_08039.md](../../sources/papers/hitch_arxiv_2611_08039.md)。

## 一句话定义

**HITCH（Harness Interfaces To Command Humanoids）是一套让 VLM 操作 Unitree G1 的 harness：感知接口返回 VLM 在图像里所选点的 3D 位置，执行接口把骨盆、躯干、双手、行走、视线命令变成全身动作；再用仿真教师在五个基础任务上的轨迹微调 VLM，使其学会「怎么用这副身体」。**（摘要、III 节）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HITCH | Harness Interfaces To Command Humanoids | 本文方法名（摘要） |
| VLM | Vision-Language Model | 视觉语言模型，本文作为智能体策略 |
| VLA | Vision-Language-Action | 视觉语言动作模型，本文的对照路线（I 节） |
| HEFT | 见 [HEFT 页](./paper-heft.md) | 本文跟踪器的训练方法，论文引用 [18] |
| SONIC | Supersizing Motion Tracking for Natural Humanoid Whole-Body Control | 对照跟踪器（论文引用 [19]） |
| AMO | 见 [AMO 页](./paper-loco-manip-161-135-amo.md) | 对照跟踪器 |
| SIMPLE | 人形移动操作仿真基准 | 主要评测基准，见 [SIMPLE 页](./paper-loco-manip-161-075-simple.md) |
| HumanoidArena | Benchmarking Egocentric Hierarchical Whole-body Learning | 留出评测基准（论文引用 [11]），无模型在其任务上训练 |
| IK | Inverse Kinematics | 规划器求解骨盆 / 躯干 / 手目标（III-B3） |
| G1 | Unitree G1 | 实验机器人，见 [Unitree G1](./unitree-g1.md) |

## 为什么重要

- **作者自述的问题：** 人形 VLA 需要大量逐任务示教，每个新任务都要全身遥操作或运动规划（I 节）；已有基础模型智能体缺少全身接口，也缺少「操作人形身体」的技能（I 节）。
- **作者自述的结果：** 不用目标基准的任务轨迹，SIMPLE 51.1%、HumanoidArena 4 任务子集 42.5%；每个 SIMPLE 任务加 5 条轨迹到 95.0%；真机 G1 无真实训练数据（Table II、IV-G）。
- **我的判断：** 它给出一种与端到端 VLA 不同的分工——VLM 低频决策、跟踪器负责平衡与动作；价值在于把「身体使用经验」做成可跨任务复用的数据，但所有结论依赖作者自改的相机与指令（见局限）。

## 核心原理

### 流程总览

```mermaid
flowchart LR
  A[语言指令加历史观察] --> B[VLM 智能体 Qwen3.8-27B]
  B -->|感知调用| C[选图像点 取深度反投影]
  C --> B
  B -->|执行调用| D[规划器 IK 加步态]
  D --> E[HEFT 跟踪器 50Hz 全身执行]
  E --> F[新头部相机图像加身体状态]
  F --> B
  G[仿真教师代码策略 五个基础任务] -->|成功轨迹| H[微调 VLM 仅对回复算 loss]
  H --> B
```

### 方法要点（均来自论文）

| 项 | 内容 | 出处 |
|----|------|------|
| 智能体循环 | 每轮一次调用（感知或执行），命令跑完才再次观察；决策以秒计 | III-A |
| 观察系 O | 原点在骨盆下方地面，每次观察重新定义；行走后旧坐标不再可靠 | III-B1 |
| 执行命令字段 | 骨盆高 0.30 至 0.79 m、躯干俯仰 / 偏航、每手 1 至 12 个路点、手指开合、走到目标或给脚步、视线（yaw ±45°，pitch 0 至 60° 向下） | III-B3、附录 A Table V |
| 跟踪器 | HEFT 训练，50 Hz；再用规划器在基础任务产生的参考继续训练，并加躯干 / 手随机外力 | III-B2 |
| 感知查询 | 1 至 100 个点，坐标 0 至 1000 归一化，按深度反投影；无深度点标无效，邻域深度差大（超过 3 cm）告警；真机用 LingBot-Depth 补全深度 | III-C、附录 A |
| 硬件 | G1 头顶加 2 自由度云台 + RealSense D455；图像 640×400，视场 90°×64° | III-C |
| 教师 | 仿真中的代码策略，只用指令和自身测量；论文称由 Claude Opus 5.5 编写；仅留成功 episode | III-D |
| 基础任务 | 抓放、双手搬箱、绕障导航、开关抽屉柜门、推拉房门；各 500 条轨迹 | III-D1、III-E |
| 微调 | 全部语言模型权重，视觉编码器冻结；loss 仅在教师回复；每个模型训练带 / 不带教师推理两个版本 | III-E、附录 C |

## 实验与结果

| 实验 | 设置 | 数字 | 出处 |
|------|------|------|------|
| 执行 | 教师代替智能体，240 个 episode，换不同跟踪器 | 教师成功率：本文跟踪器 97.9%，HEFT 94.2%，SONIC 27.9%，AMO 12.9%；手部误差 0.6 至 0.7 cm，步行误差 SIMPLE 5%、HumanoidArena 3% | IV-B Table I |
| 命令消融 | 去掉躯干俯仰 / 偏航；去掉手掌朝向 | 教师成功率 67.1%；18.8% | Table I |
| 零任务轨迹迁移 | 仅基础任务 | SIMPLE 3.3%（预训练）到 51.1%；HumanoidArena 4 任务子集 16.3% 到 42.5% | IV-C Table II |
| 少量轨迹 | 每任务 N 条 SIMPLE 轨迹，单独训练 / 加基础任务 | N=5：90.0% / 95.0%；N=10：95.0% / 98.9%；N=30：96.7% / 98.9% | IV-D Table II |
| 去掉感知 | 训练和接口都删去感知调用 | 基础任务模型 SIMPLE 2.8%，HumanoidArena 27.5%；加 10 条 SIMPLE 轨迹仍 80.6%（有感知 98.9%） | IV-E |
| 推理 | 带 / 不带教师推理 | 微调后 SIMPLE 不带推理更高（如基础 + 10 条：98.9% 对 89.4%）；HumanoidArena 无一致优势 | IV-F Table III |
| 大模型对照 | Qwen3.8-Max API 不微调 | SIMPLE 12.2%（不思考）/ 7.8%（思考） | Table III |
| 真机 G1 | 仅基础任务，不带推理，橡胶手 | 搬箱 13/20（朝桌 8/10，朝折叠桌 5/10），导航 6/10 | IV-G |

评测规模：SIMPLE 每项结果 180 个 episode（6 任务各 30），HumanoidArena 每项 80 个（4 任务各 20）（IV-A）。

## 工程实践

- **接口设计原则（作者）：** 智能体决策频率低，所以要「先测量再行动」，并要求命令被准确执行；教师的每个动作必须能由学生可见的观察推出，避免教师读仿真真值造成模仿失败（III-A）。
- **指令写法（附录 E Table VI）：** 论文重写了官方指令，写入「每个 episode 都成立的事实」（如篮子在身后桌上、货架在左侧），但不给具体位置。复现或对比时需注意这一点。
- **耗时（IV-G Table IV）：** 成功真机 episode 搬箱平均 60.7 s，执行占 61%（其中运动 44%），感知 20%，决策 19%；作者结论是加快图像采集和规划比换更快的模型收益更大。
- **部署：** 模型在远程服务器上，通过 Wi-Fi 与机器人收发观察和命令（IV-G）。
- **开源：** 论文正文没有代码链接；项目页 [hitch.axell.top](https://hitch.axell.top) 静态抓取为空壳，开源状态未确认。

## 局限

**论文自述（V 节、IV 节）：**

- 规划器不支持坐下、跨越障碍等动作；HumanoidArena 因此只用了 4 个任务（IV-A）。
- 加入 SIMPLE 轨迹会降低 HumanoidArena 成功率（42.5% 到 28.8%），一个基准的数据会窄化基础任务所教的内容（IV-C）。
- 微调后加推理反而降低 SIMPLE 成功率，主要因为对错误读数不加质疑；教师几乎不犯感知 / 推理错，示范里没有纠错（IV-F）。
- 命令跑完才再观察，执行中不能反应；真机手没有手指控制，只能做两个任务（V）。

**我看到的未验证之处（判断，非论文自述）：**

- 两个基准的相机都换成了自己的云台相机，指令也被重写，所以 51.1% / 42.5% 不能直接和其他方法在官方设置下的数字比较；成功判据时限也比官方长（附录 F Table VII，如 SIMPLE BendPick 官方 8 s，对应平均 16.3 s）。
- 真机只有两个任务、20 与 10 个 episode，且搬箱两个方向结果 8/10 对 5/10 差距明显，样本小（IV-G）。
- 论文未与端到端 VLA 在同一设置下做正面对比（我只在文中看到相关工作层面的论述，II-B），「优于 VLA」不能从本文直接读出（推测：此对比需另行复现）。
- 教师由 LLM 编写代码、且依赖仿真，成本与可迁移性（到更多任务）论文未量化。

## 关联页面

- [VLA 方法页](../methods/vla.md)：HITCH 的对照路线（端到端动作模型）
- [全身控制](../concepts/whole-body-control.md)
- [Loco-manipulation 任务](../tasks/loco-manipulation.md)
- [HEFT](./paper-heft.md)：跟踪器训练方法
- [SONIC（NVIDIA）](./nvidia-sonic.md) 与 [AMO](./paper-loco-manip-161-135-amo.md)：对照跟踪器
- [SIMPLE 基准](./paper-loco-manip-161-075-simple.md)
- [Psi0](./paper-loco-manip-161-156-psi0.md)：论文引用的人形基础模型
- [VoxPoser](./paper-voxposer.md)：同属「语言模型驱动 3D 价值图 / 代码」路线
- [Qwen3-VL](./qwen3-vl.md)：论文沿用其归一化坐标 grounding 约定
- [Unitree G1](./unitree-g1.md)

## 参考来源

- [sources/papers/hitch_arxiv_2611_08039.md](../../sources/papers/hitch_arxiv_2611_08039.md)：通读归档与周报核对
- [arXiv:2611.08039](https://arxiv.org/abs/2611.08039)

## 推荐继续阅读

- [arXiv HTML 全文](https://arxiv.org/html/2611.08039v1)
- [项目页](https://hitch.axell.top)
- [HEFT 页](./paper-heft.md)
