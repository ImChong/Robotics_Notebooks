---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-learning-affordance-landscapes-for-interaction-explorati
url: "https://proceedings.neurips.cc/paper_files/paper/2020/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html"
venue: "NeurIPS 2020"
summary: "NeurIPS 2020 的 affordance 论文，清单分组：Perception（Scope: Scene；Perception Subcategory: Spatial）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_learning-affordance-landscapes-for-interaction-explorati.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# IntExp

**Learning Affordance Landscapes for Interaction Exploration in 3D Environments**（NeurIPS 2020）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Learning Affordance Landscapes for Interaction Exploration in 3D Environments——在清单中归入 **Perception（Scope: Scene；Perception Subcategory: Spatial）**（清单未附方法摘要，此处不替原文概括贡献）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图](../overview/affordance-awesome-technology-map.md) 的同分组条目横向阅读同方向工作。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | NeurIPS 2020 |
| 方法名（清单标注） | IntExp |
| 分组 1 | Perception · Scope: Scene；Perception Subcategory: Spatial |
| 论文 | <https://proceedings.neurips.cc/paper_files/paper/2020/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html> |

## 核心机制（归纳）

> 注：以下机制与评测指标基于通读开放获取原文全文核对补充。

### 问题定义与 POMDP 探索目标

`IntExp` 提出**面向交互的探索（Exploration for Interaction）**问题：具身智能体进入未建图的陌生 3D 室内环境（如新厨房），在不知物体类别与动作效果的前提下，需自主移动并尝试交互，以尽快覆盖环境中所有可交互物体与有效动作，并同步建立视觉可供性地形图（Affordance Landscape）。

- **观测与动作空间**：每步观测为 `(x_t, theta_t)`（第一人称 RGB-D 图像与里程计）；动作空间 $A = A_N \cup A_I$ 包含导航动作 $A_N = \{\text{move forward}, \text{turn left/right } 30^\circ, \text{look up/down } 15^\circ\}$ 与作用于视野中心的交互动作 $A_I = \{\text{take}, \text{put}, \text{open}, \text{close}, \text{toggle-on}, \text{toggle-off}, \text{slice}\}$。
- **新交互计数奖励**：$R(s_t, a_t, s_{t+1}) = 1$ 当且仅当 $a_t \in A_I$ 且 $c(a_t, o_t) = 0$（首次成功对物体 $o_t$ 执行动作 $a_t$），否则为 $0$，激励智能体广度覆盖不同物体–动作对而非重复刷单一交互。

### 在线可供性学习与策略协同训练

1. **交互轨迹反投影打标（Backprojected Markers）**：在策略 $\pi_E$ 采样的轨迹 $\tau$ 中，每当发生交互 $a_t \in A_I$，利用深度与里程计通过逆透视投影计算视野中心 3D 世界坐标 $p_t$，连同动作类型与成功标志 $z_t \in \{0, 1\}$ 存入记忆 $M = \{(p_t, a_t, z_t)\}$。回合结束后，将记忆中的三维标记重投影到轨迹各帧，像素在同类标记 $\delta = 20\text{cm}$ 欧氏距离内标为 $1$（成功）或 $0$（失败），未覆盖区域标为 $-1$（未知），形成 $|A_I| \times H \times W$ 稀疏/含噪掩码。原文对比两种标记粒度：固定尺度单点标记 `IntExp(Pt)`（默认，跨未见场景泛化更稳）与目标物体整物标记 `IntExp(Obj)`。
2. **双头 U-Net 可供性分割模型 `F_A`**：输入 RGB-D 图像，输出成功概率头 $\hat{y}_A$（仅在 $y_{ij} \neq -1$ 像素上计算交叉熵）与可交互不确定性头 $\hat{y}_I$（全图预测 $1[y = -1]$，用于抑制天花板、窗户等极少出现标记的区域）。联合损失 $L = L_{\text{ce}}(\hat{y}_A, y, \forall y_{ij}\neq -1) + L_{\text{ce}}(\hat{y}_I, 1[y=-1], \forall y_{ij})$，最终可供性得分为 $\hat{y} = \hat{y}_A \times (1 - \hat{y}_I)$。
3. **Actor-Critic 策略与下游迁移**：当前 RGB 帧 $x$ 与预测可供性图 $\hat{y} = F_A(x)$ 各自经过 3 层 CNN 编码后拼接并经 FC 融合，送入 **GRU** 聚合时序历史，再经 Actor-Critic 线性层以 **PPO**（1M 帧，rollout `T = 256`）训练。策略 $\pi_E$ 与分割模型 $F_A$ 交替迭代更新。迁移至下游多步稀疏奖励任务时，冻结 CNN、GRU 与 $F_A$ 参数，仅微调 Actor-Critic 线性层（500k 帧 PPO）。

### 方法流程图

```mermaid
flowchart LR
    OBS["第一人称 RGB-D 观测与里程计 (x_t, theta_t)"] --> UNET["U-Net 可供性模型 F_A (双头输出 y_A 与不确定性 y_I)"]
    UNET --> MAP["可供性图 y_hat = y_A * (1 - y_I)"]
    OBS --> CNN1["3层 CNN 图像编码"]
    MAP --> CNN2["3层 CNN 可供性编码"]
    CNN1 --> FUSE["特征拼接 + FC + GRU 时序聚合"]
    CNN2 --> FUSE
    FUSE --> PPO["Actor-Critic 策略 pi_E (PPO 1M 帧训练)"]
    PPO --> ENV["AI2-iTHOR 交互执行与计数奖励 R"]
    ENV --> MEM["逆透视投影记忆 M = {(p_t, a_t, z_t)}"]
    MEM --> MASK["跨帧重投影构建伪掩码 (delta=20cm)"]
    MASK --> UNET
```

## 评测与指标

- **实验环境**：**AI2-iTHOR** 全部 30 个厨房场景（69 类物体，每类 1–5 种交互；划分 20 训练 / 5 验证 / 5 测试，随机化物体位置、初始状态如 `isOpen`/`isToggled` 及智能体初始位姿）。每个测试场景评估 80 个随机 episode（每轮 1024 步），相对最短路径 Oracle 计算交互覆盖率 **Coverage** 与尝试准确率 **Precision**。
- **交互探索性能（Table 1 & Sec. 4.1）**：
  - 平均 Precision / Coverage（%）：`Random` **2.39 / 9.81**、`Curiosity` **2.26 / 9.84**、`Novelty` **3.39 / 14.50**、`Random+` **3.53 / 11.89**、`ObjCoverage` **4.15 / 16.77**、**`IntExp`** **8.46 / 22.15**（分项：`Take` 8.53/19.23、`Put` 6.35/39.62、`Open` 10.18/30.05、`Close` 9.82/24.43、`Turn-on` 12.42/23.19、`Turn-off` 11.91/18.45、`Slice` 0.04/0.06）。
  - 探索效率：`IntExp` 发现的独特交互数是最佳基线的 **1.33×**，并在 **42% 更少时间步**内达到最佳探索基线的水平（较 `Random+` 少 **63%** 步数；在 `T = 256` 早期阶段交互发现数是 `Novelty` 的 **2.5×**）。
  - 视觉先验消融（Fig. 3 右）：`IntExp(Pt)` 覆盖率达 **0.252**，高于 `IntExp(Obj)`（**0.236**，因整物掩码引入较多噪声）、显著性先验 `IntExp(Sal)` 与纯视觉 `IntExp(RGB)`，逼近使用仿真真值掩码的 `IntExp(GT)`。
- **视觉可供性预测（Fig. 5，23,637 个未见测试视角，mAP）**：`All-Ones` **8.9**、`IntExp(Sal)` **13.4**、`IntExp(Pt)` **16.3**、`IntExp(Obj)` **26.5**（`IntExp(GT)` 上界 **37.3**）。
- **下游多步任务微调（Fig. 6，400 测试 episode 成功率 %，无人类示教、仅子目标 +10 奖励下 PPO 微调 500k 帧）**：
  - `Retrieve`（从柜/抽屉取出物体放置）：`IntExp` **27.25%** vs `ObjCoverage` 18.50%、`Scratch` 8.75%、`Novelty` 8.00%、`Curiosity` 2.75%
  - `Store`（将物体放入柜/抽屉并关门）：`IntExp` **27.00%** vs `ObjCoverage` 10.75%、`Novelty` 6.50%、`Scratch` 6.25%
  - `Wash`（物体放入水槽并打开水龙头）：`IntExp` **3.75%** vs `Curiosity` 1.50%、`Scratch` 1.25%、`ObjCoverage` 0.75%
  - `Heat`（锅具放上灶台并打开炉灶）：`IntExp` **19.00%** vs `Curiosity` 2.25%、`Scratch` 1.75%、`Novelty` 1.25%、`ObjCoverage` 0.50%
  - 总体上，预训练交互探索策略与可供性模型使下游任务成功率最高提升 **16 个百分点**。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**`IntExp` 将自主交互探索与在线可供性分割闭环结合：智能体通过探索产生稀疏交互标记反哺像素级可供性预测，再利用可供性图引导高效探索并加速下游长程操作任务学习。**

- **双头不确定性建模有效**：通过 $\hat{y}_I$ 显式区分“交互失败”与“未曾尝试（$-1$）”，缓解了探索初期自监督标注稀疏与偏置问题。
- **跨步复杂交互仍是瓶颈**：像 `Slice`（需先找到并拿起刀再切物体，各方法覆盖率均 `< 1%`）与 `Wash`（`3.75%`）等强依赖历史持有状态的长链交互仍极易被探索策略忽略，原文指出未来需在可供性模型中引入手持状态与空间记忆。
- **项目与源码**：论文给出项目主页 <http://vision.cs.utexas.edu/projects/interaction-exploration/>，实际代码开源状态请以该页为准。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_learning-affordance-landscapes-for-interaction-explorati.md`](../../sources/papers/affordance_awesome_learning-affordance-landscapes-for-interaction-explorati.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://proceedings.neurips.cc/paper_files/paper/2020/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html>

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://proceedings.neurips.cc/paper_files/paper/2020/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
