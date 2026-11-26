---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
venue: "CVPR 2025"
summary: "RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_ref_4aab09abc95513cc8eb4_robotwin-dual-arm-robot-benchmark-with-g.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# RoboTwin

**RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 051/564** 篇，分组 **Benchmarks & simulators**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Benchmarks & simulators）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 051/564 |
| 分组 | Benchmarks & simulators |
| 出处 | CVPR 2025 |
| 论文 | <https://openaccess.thecvf.com/content/CVPR2025/papers/Mu_RoboTwin_Dual-Arm_Robot_Benchmark_with_Generative_Digital_Twins_CVPR_2025_paper.pdf> |
| 项目页 | <https://robotwin-benchmark.github.io> |
| 子类 / 象限 | 机器人操作基准 · 物理仿真 · 合成数据与数据生成 · 不适用 |

## 核心机制（归纳）

> 注：以下机制与评测指标基于通读开放获取原文全文核对补充（[CVPR 2025 原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Mu_RoboTwin_Dual-Arm_Robot_Benchmark_with_Generative_Digital_Twins_CVPR_2025_paper.pdf)）。

**RoboTwin** 是一套结合 3D 生成式基础模型与大语言模型空间推理代码生成的双臂机器人操作基准与 Sim-to-Real 数据合成框架，并与开源双臂硬件平台 **COBOT Magic** 严格对齐（与 [World Action Models（WAM）](../concepts/world-action-models.md) 的具身操作训练与评测管线密切相关）：

1. **单图驱动的多样化数字孪生资产生成（§3.1）**：
   - 从真实物体单张 2D RGB 图像出发，先由 **GPT-4V** 生成物体文本描述并经语言模型改写为多样外观变体，再用 **SDXL-Turbo** 合成多外观 2D 图像集，最后输入 Deemos **Rodin** 图像条件 3D 生成模型重建带几何、法线、线框与纹理的 3D 资产。
   - **双重质检与物理随机化**：结合 **UCLIP-I** 定量相似度与 **GPT-4V** 定性视觉核验自动重生成低质资产，并由 GPT-4V 推断材质类别后赋予带 $\pm 5\%$ 随机扰动的物理参数。
2. **工具中心空间标注与跨实例关键点迁移（§3.2）**：
   - 为每类工具标注两类关键点（功能点 `Point for Function`、接触点 `Point for Contact`）及三正交主轴（功能轴 `Function Axis`、接近轴 `Approach Axis`、侧向轴 `Lateral Axis`）。
   - 利用 **Stable Diffusion** 稠密扩散特征在桌面俯视视角下做像素级特征点匹配，将源模型标注零重标地迁移至同类其余生成 3D 实例。
3. **空间关系感知的双臂专家轨迹代码生成（§3.3）**：
   - 双臂系统支持螺旋运动插值与夹爪同步协同、非对称独立操作及动态臂间避碰。
   - 流程分六步：场景初始化（含可达性校验） $\rightarrow$ LLM 任务分解 $\rightarrow$ 层次化空间/时序约束推断（如抓取接触点与接近轴对齐、功能点与目标几何约束对齐） $\rightarrow$ LLM 调用预定义 API 并基于 **MPlib** 求解受运动学 $T_{\text{ee}} = f_{\text{FK}}(\theta(t))$、位置/朝向对齐与无碰撞空间 $\theta(t) \in \mathcal{C}$ 约束的轨迹优化问题 $\rightarrow$ 成功判定 $\rightarrow$ 将运行时报错、规划失败与末态偏差反馈给 LLM 迭代自修正。
4. **真机–仿真对齐双臂基准（§4）**：
   - 基于 **ManiSkill3 / SAPIEN** 构建 **15** 项双臂任务（基准表评测 14 项），硬件严格对应 **COBOT Magic**（Tracer AGV 底盘 + 4 条 AgileX 机械臂含主从双臂 + 4 台 30 Hz Intel RealSense D435 RGB-D 相机），每项任务提供 100 组仿真数据与 20 组真机遥操作数据（含 RGB-D、彩色/无色点云及主从臂关节/末端位姿）。

### 方法流程图

```mermaid
flowchart LR
  RGB["Single Real 2D RGB Image"] --> VLM["GPT-4V Desc + LLM Variations + SDXL-Turbo"]
  VLM --> Rodin["Rodin 3D Generation + UCLIP-I / GPT-4V Check"]
  Rodin --> Annot["Spatial Annotation (Function/Contact Points + 3 Axes)"]
  Annot -->|"SD Feature Matching"| Transfer["Cross-Instance Keypoint Transfer"]
  Transfer --> LLMPlan["LLM Task Decomposition & Constraint Inference"]
  LLMPlan --> MPlib["MPlib Constrained Trajectory Opt (FK + Collision-Free)"]
  MPlib --> EvalLoop["Execution Check & Error Feedback Self-Correction"]
  EvalLoop -->|"Pass"| SimData["RoboTwin Sim Demos (ManiSkill3 / SAPIEN)"]
  SimData -->|"Pretrain 300 Sim + Finetune 20 Real"| Policy["IL Policy Deployment (COBOT Magic Dual-Arm)"]
```

## 评测与指标

- **仿真模仿学习基准评测（14 项任务，D435 相机设定，分别使用 20 / 50 / 100 条专家示范、3 随机种子，Table 1）**：
  - 对比 2D **Diffusion Policy（`DP`）** 与 3D 点云 **3D Diffusion Policy（`DP3 (XYZ)` / `DP3 (XYZ+RGB)`）**，代表性结果（20 $\rightarrow$ 50 $\rightarrow$ 100 示范成功率 %）：
    - **`Dual Bottles Pick (Easy)`**：`DP` 从 1.7±0.6 $\rightarrow$ 38.3±6.7 $\rightarrow$ **85.7±6.7**（随示范量增长显著超越 `DP3 (XYZ)` 的 55.3±11.5 和 `DP3 (XYZ+RGB)` 的 75.7±17）。
    - **`Container Place`**：`DP3 (XYZ)` 达 52.7±5.0 $\rightarrow$ 77.7±2.5 $\rightarrow$ **85.3±3.2**，而引入颜色后 `DP3 (XYZ+RGB)` 降至 37.3 $\rightarrow$ 51.3 $\rightarrow$ 62.3，说明点云与 RGB 特征融合尚不成熟。
    - **`Pick Apple Messy`（杂乱桌面）**：`DP3 (XYZ+RGB)` 在 100 示范下达 **54.0±12.8**，显著高于 `DP3 (XYZ)` 的 9.7±2.1 与 `DP` 的 29.3±5.0。
    - **`Put Apple Cabinet` / `Shoe Place`**：100 示范下 `DP3 (XYZ+RGB)` 在 Put Apple Cabinet 达 **78.3±3.8**（`DP` 8.0±12.2）；`DP3 (XYZ)` 在 Shoe Place 达 **54.3±0.6**（50 示范为 59.3±7.4，`DP` 仅 33.0±15.8）。
    - **高难强协同任务仍遇瓶颈**：`Mug Hanging (Hard)` 最优 `DP3 (XYZ)` 仅 **15.3±5.5**（`DP` 0.0）；`Dual Shoes Place` 所有方法在 100 示范下均低于 **15%**（`DP3 (XYZ)` 12.0±1.7、`DP3 (XYZ+RGB)` 6.0±1.0、`DP` 3.0±1.0）。
- **真机 Sim-to-Real 迁移评测（COBOT Magic，每任务 50 次随机初始位姿试验，Table 2 & Table 3）**：
  - 对比「仅用 20 条真机数据训练」与「300 条 RoboTwin 仿真预训练 + 20 条真机微调」：
  - **单臂任务（Table 2，5 项任务）**：平均成功率由 **1.2%** 跃升至 **72%**（提升超 70 个百分点）——`Bottle Pick (Easy)` 0/50 $\rightarrow$ **42/50**、`Bottle Pick (Hard)` 0/50 $\rightarrow$ **16/50**、`Container Place` 0/50 $\rightarrow$ **49/50**、`Cup Place` 1/50 $\rightarrow$ **39/50**、`Hammer Beat` 2/50 $\rightarrow$ **37/50**。
  - **双臂任务（Table 3，5 项任务）**：平均成功率由 **20%** 提升至 **62%**（提升超 40 个百分点）——`Dual bottle Pick (Easy)` 0/50 $\rightarrow$ **31/50**、`Dual bottle Pick (Hard)` 0/50 $\rightarrow$ **11/50**、`Container Place` 25/50 $\rightarrow$ **44/50**、`Cup Place` 0/50 $\rightarrow$ **26/50**、`Sweep Ball` 25/50 $\rightarrow$ **43/50**。
  - **真机数据替代效应（Fig. 7）**：`300 Sim + 20 Real` 在 Bottle Pick 与 Cup Place 上取得与纯采集 **300 条真机数据** 相当的成功率。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Benchmarks & simulators** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**RoboTwin 将「单图 3D 生成数字孪生 + 扩散特征跨实例标注迁移 + LLM 空间约束代码生成」串联为可规模化的双臂仿真数据工厂，使 300 条合成示范配合仅 20 条真机微调即可在 COBOT Magic 上获得接近 300 条纯真机数据的迁移效果（单臂 +70%+、双臂 +40%+）。**

- **样本效率互补**：仿真基准表明 `DP3` 在 20 条少样本下几何先验强，但 100 条规模时 2D `DP` 扩展性更好；同时 `DP3(XYZ+RGB)` 在杂乱场景受益明显但在干净几何任务反而退化，揭示 3D 点云与 RGB 语义融合仍待改进。
- **低成本突破真机瓶颈**：通过高对齐硬件建模与合成数据预训练，单臂真机平均成功率从 1.2% 提升至 72%、双臂从 20% 提升至 62%。
- **双臂强协同仍存显著短板**：两臂动作空间可解耦且按距离分区完成的任务成功率较高，但像 `Dual Shoes Place` 这类需强时空耦合/闭链协同的任务，现有模仿学习算法成功率仍低于 15%；倒卧瓶子等复杂初始姿态泛化也仍有明显提升空间。
- 项目主页见 <https://robotwin-benchmark.github.io>，对应代码、仿真器与数据集入口以项目页发布为准。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_ref_4aab09abc95513cc8eb4_robotwin-dual-arm-robot-benchmark-with-g.md`](../../sources/papers/rcl_awesome_wam_ref_4aab09abc95513cc8eb4_robotwin-dual-arm-robot-benchmark-with-g.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://openaccess.thecvf.com/content/CVPR2025/papers/Mu_RoboTwin_Dual-Arm_Robot_Benchmark_with_Generative_Digital_Twins_CVPR_2025_paper.pdf>

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Mu_RoboTwin_Dual-Arm_Robot_Benchmark_with_Generative_Digital_Twins_CVPR_2025_paper.pdf)
