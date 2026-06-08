# OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction（arXiv:2509.26633）

> 来源归档（ingest · 全文消化）

- **标题：** OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction
- **类型：** paper / humanoid motion retargeting + data generation + loco-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2509.26633>
- **arXiv HTML：** <https://arxiv.org/html/2509.26633v1>
- **PDF：** <https://omniretarget.github.io/static/images/paper.pdf>（项目页镜像）；<https://arxiv.org/pdf/2509.26633>
- **项目页：** <https://omniretarget.github.io/>（归档见 [`sources/sites/omniretarget-github-io.md`](../sites/omniretarget-github-io.md)）
- **会议：** ICRA 2026（项目页标注）
- **代码：** <https://github.com/amazon-far/holosoma>（归档见 [`sources/repos/holosoma.md`](../repos/holosoma.md)）
- **数据集：** <https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset>（归档见 [`sources/sites/omniretarget-dataset-huggingface.md`](../sites/omniretarget-dataset-huggingface.md)）
- **作者：** Lujie Yang*, Xiaoyu Huang*, Zhen Wu*, Angjoo Kanazawa†, Pieter Abbeel†, Carmelo Sferrazza†, C. Karen Liu†, Rocky Duan†, Guanya Shi†（* equal；† Amazon FAR co-lead）
- **机构：** Amazon FAR、MIT、UC Berkeley、Stanford、CMU
- **硬件：** Unitree G1（主实验）；亦展示 G1 / H1 / Booster T1 跨 embodiment 重定向
- **入库日期：** 2026-06-08（深化；初版 2026-05-31）
- **一句话说明：** 基于 **interaction mesh** 的交互保留重定向引擎：逐帧 **Sequential SOCP** 最小化人–机 mesh 的 **Laplacian 形变能** 并满足**硬约束**（SDF 非穿透、关节/速度界、stance 脚粘附），支持单演示到多 embodiment / 地形 / 物体的**数据增广**；下游 RL 仅用 **5 项 reward + 4 项 DR**、**无 curriculum** 即可在 G1 上零样本实机执行长达 **30 s** 的 parkour / loco-manipulation。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://omniretarget.github.io/> | 真机视频、增广交互演示、GMR/PHC 基线对比、LAFAN1 可视化 |
| 代码 | <https://github.com/amazon-far/holosoma> | 训练 + 推理 + `holosoma_retargeting`（OmniRetarget 实现） |
| 数据集 | <https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset> | G1 重定向轨迹 4.0 h（OMOMO + 自采 MoCap；LAFAN1 需自行重定向） |
| 下游跑酷 | [PHP（arXiv:2602.15827）](https://arxiv.org/abs/2602.15827) | 用 OmniRetarget 构建 G1 **原子跑酷技能库** → motion matching 长程合成 |
| 极简 tracking 先例 | [BeyondMimic（arXiv:2508.08241）](https://arxiv.org/abs/2508.08241) | 干净参考下 **5 项 reward** 即可 SOTA 真机 tracking；OmniRetarget 将同一叙事扩展到 **交互场景** |
| 图形学近邻 | IMMA [22] | 同样用 interaction mesh，但**未开源**、无环境/物体交互与硬约束全集 |
| 基线 | PHC [10]、GMR [9]、VideoMimic [11] | 关键点匹配或软惩罚；缺交互保留与数据增广 |

## 摘要级要点

- **痛点：** 常见重定向产生脚滑、穿透，且**不显式保留**人–物–地形交互，导致参考质量差、下游 RL 需大量 ad-hoc reward。
- **方法：** Delaunay 四面体 **interaction mesh** 连接关键关节 + 物体/环境采样点；每帧解 **Sequential SOCP**，目标为 Laplacian 坐标差最小 + 时间平滑 $Q$，约束含 SDF 非穿透、关节/速度界、stance 脚位置固定。
- **增广：** 固定源 demonstration mesh $\mathcal{P}_t^{\text{source}}$，变化目标物体位姿/形状或地形高度，在物体局部系建 mesh；下身锚定 nominal 轨迹防 trivial 刚体漂移。
- **实现：** **Drake** 自动微分处理四元数浮动基在 $\mathbb{S}^3$ 上的导数；支持 **G1 / H1 / T1** 仅改关键点对应与碰撞模型。
- **下游 RL：** 最小本体感知观测 + **5 reward**（body/object DeepMimic tracking、action rate、软关节限、自碰）+ **4 DR**；与 BeyondMimic [33] 超参 **开箱即用**。
- **规模：** OMOMO、LAFAN1、自采 MoCap → 论文 **8+ 小时**、项目页 **9+ 小时** 重定向；HF 公开 **4.0 h** 子集。

## 核心摘录（面向 wiki 编译）

### 1) 优化问题（每帧）

$$\min_{q_t} \sum_i \|L(p_{t,i}^{\text{source}}) - L(p_{t,i}^{\text{target}}(q_t))\|^2 + \|q_t - q_{t-1}\|_Q^2$$

约束：SDF 非穿透 $\phi_j(q_t) \geq 0$；关节界 $q_{\min} \leq q_t \leq q_{\max}$；速度界；stance 脚 $p_t^F = p_{t-1}^F$（源运动 xy 速度 < 1 cm/s 判为支撑相）。

### 2) 与基线对比（Table I 维度）

| 方法 | 硬运动学约束 | 物体交互 | 地形交互 | 数据增广 | 优化 |
|------|-------------|----------|----------|----------|------|
| IMMA | ✓ | ✗ | ✗ | ✗ | QP |
| PHC / GMR | ✗ | ✗ | ✗ | ✗ | GD / Mink |
| VideoMimic | 软惩罚 | ✗ | ✓ | ✗ | JAX L-M |
| **OmniRetarget** | ✓ | ✓ | ✓ | ✓ | Sequential SOCP |

### 3) Kinematic 质量（Table II 摘要 · OMOMO 物体交互）

| 方法 | 穿透时长 ↓ | 最大深度 (cm) ↓ | 脚滑时长 ↓ | 接触保留 ↑ | 下游 RL 成功率 ↑ |
|------|-----------|----------------|-----------|------------|----------------|
| PHC | 0.68±0.21 | 5.11±3.09 | 0.05±0.05 | 0.96±0.09 | 71.3%±22.6% |
| GMR | 0.83±0.14 | 8.50±3.94 | 0.02±0.01 | 0.99±0.04 | 50.8%±23.9% |
| VideoMimic | 0.60±0.27 | 7.48±4.95 | 0.12±0.07 | — | — |
| **OmniRetarget** | **更低**（全文 Table II） | **更低** | **更低** | **更高** | **更高** |

### 4) 下游 RL 配方

**观测（最小本体感知）：** 参考关节位/速、骨盆位姿误差；本体骨盆线/角速度、关节位/速；上一步动作。敏捷动作可 mask 骨盆线位置误差与速度。

**5 项 reward：** body tracking（DeepMimic 式位姿/速度）、object tracking（适用时）、action rate、soft joint limit、self-collision（>1 N 二值惩罚）。

**4 项 DR（机器人）：** 躯干 COM ±(0.025, 0.05, 0.075) m；关节默认位 ±0.01 rad；随机推 0.3 m/s / 0.78 rad/s (1–3 s)；观测噪声（Rot6D 姿态、速度、关节）。

**物体 DR：** 质量 0.1–2 kg、COM ±0.08 m、惯量 50–150%、形状 ±10%。

### 5) 旗舰真机演示

- **30 s 长程：** 搬运 4.6 kg 椅子 → 作踏台攀 0.9 m 高台 → 跳下并翻滚缓冲（Atlas 工具使用 demo 灵感）。
- **墙翻：** ~0.5 s 完成；峰值角速度 **15 rad/s**；需 IMU 量程 >15 rad/s；训练时放宽末端误差阈值至 0.5 m。
- **增广评估：** 全增广集训练成功率 **79.1%** vs 仅 nominal 评估 **82.2%**；纯 DR 扰动物体位姿/形状效果差。

### 6) 与 PHP 的关系

- **PHP [43] 明确使用 OmniRetarget** 将人类跑酷 MoCap 转为 G1 **原子技能库**，再 motion matching 长程合成。
- 同一 Amazon FAR 团队在 **2025 重定向数据层** 与 **2026 感知跑酷策略层** 形成衔接。

## 对 wiki 的映射

- 深化实体页：[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)
- 代码实体：[`wiki/entities/holosoma.md`](../../wiki/entities/holosoma.md)
- 数据集实体：[`wiki/entities/omniretarget-dataset.md`](../../wiki/entities/omniretarget-dataset.md)
- 交叉：[`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[`wiki/concepts/motion-retargeting.md`](../../wiki/concepts/motion-retargeting.md)、[`wiki/tasks/loco-manipulation.md`](../../wiki/tasks/loco-manipulation.md)

## 关联原始资料

- 项目页：[`sources/sites/omniretarget-github-io.md`](../sites/omniretarget-github-io.md)
- 代码：[`sources/repos/holosoma.md`](../repos/holosoma.md)
- 数据集：[`sources/sites/omniretarget-dataset-huggingface.md`](../sites/omniretarget-dataset-huggingface.md)
- 42 篇栈策展：[`humanoid_rl_stack_03_omniretarget_interaction_preserving_data_generat.md`](humanoid_rl_stack_03_omniretarget_interaction_preserving_data_generat.md)
