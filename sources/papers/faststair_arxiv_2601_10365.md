# FastStair：Learning to Run Up Stairs with Humanoid Robots（arXiv:2601.10365）

> 论文来源归档（ingest）

- **标题：** FastStair: Learning to Run Up Stairs with Humanoid Robots
- **类型：** paper / humanoid locomotion / reinforcement learning + model-based foothold guidance
- **arXiv：** <https://arxiv.org/abs/2601.10365>（HTML：<https://arxiv.org/html/2601.10365v1>）
- **PDF：** <https://arxiv.org/pdf/2601.10365.pdf>
- **机构：** 哈尔滨工业大学、逐际动力（LimX Dynamics）、ZJUI、USTC、HKUST、NUS 等（论文标注主要工作在 LimX Dynamics）
- **项目页：** <https://npcliu.github.io/FastStair>
- **入库日期：** 2026-05-17
- **一句话说明：** 用 **GPU 并行离散搜索的 DCM 落脚点规划** 在 Isaac Lab 大规模 RL 中提供 **显式可行落脚点监督**，先训出 **偏安全/保守的基策略**，再通过 **高速/低速专家分化 + LoRA 融合** 缓解规划器保守性与全速域动作分布差异，在 **LimX Oli** 上实现 **指令速度至约 1.65 m/s** 的稳定上楼梯与长螺旋梯实机演示。

## 核心论文摘录

### 1) 问题设定：敏捷 vs 稳定、无模型 RL vs 硬约束规划

- **要点：** 楼梯场景同时要求 **高敏捷** 与 **严格稳定**；纯 model-free RL 常靠 **隐式稳定性奖励与任务相关 reward shaping**，在楼梯上易出现 **不安全行为**；纯 model-based foothold 规划把 **接触可行性与平衡结构** 编码为硬约束，又常导致 **保守步态、限制速度**。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-faststair-humanoid-stair-ascent.md`](../../wiki/entities/paper-faststair-humanoid-stair-ascent.md)
  - [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)

### 2) 并行 DCM 落脚点规划 +「优化改写为离散搜索」

- **要点：** 采用 **变高倒立摆（VHIP）** 描述上楼时的 CoM 高度斜坡，推导楼梯上的 **DCM（Divergent Component of Motion）** 演化；将 foothold 优化改写为在离散候选集上的 **并行代价评估 + argmin**，用 **张量化 GPU 运算** 嵌入并行 RL，论文报告相对「把实时优化硬塞进训练」类路线约 **25×** 量级的训练加速叙事（以论文实验段落为准）。
- **对 wiki 的映射：**
  - [`wiki/concepts/capture-point-dcm.md`](../../wiki/concepts/capture-point-dcm.md)
  - [`wiki/concepts/footstep-planning.md`](../../wiki/concepts/footstep-planning.md)

### 3) 三阶段 RL：规划引导预训练 → 速度专家 → LoRA 统一策略

- **要点：** **阶段 1** 用规划器给出的最优落脚点驱动 **foothold-tracking reward**，优先 **穿越稳定性**；**阶段 2** 调整奖励权重并划分 **低速 / 高速** 指令子区间，分别从基策略微调出 **两个专家**（论文指出单策略直接覆盖全速域易出现 **向中等速度塌缩**）；**阶段 3** 将两专家参数并入 **单一网络**，在分支上使用 **LoRA** 微调以消除硬切换带来的 **控制不连续 / 抖振**，部署时用 **规则切换器** 按指令速度选择专家分支。
- **对 wiki 的映射：**
  - [`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)
  - [`wiki/concepts/privileged-training.md`](../../wiki/concepts/privileged-training.md)

### 4) 仿真与实机部署要点（归纳）

- **要点：** Isaac Lab **4096** 并行环境；本体观测含 **机载局部高程图**；特权观测含规划器 **最优落脚点** 等；实机 **RealSense D435i** 重建约 **1.8 m × 1.2 m、5 cm 分辨率** 地形网格；报告 **33 级螺旋梯（每级约 17 cm 踢面高度）约 12 s** 完成，以及 **广州塔机器人跑楼赛** 冠军叙事（以论文与新闻口径为准）。
- **对 wiki 的映射：**
  - [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)
  - [`wiki/entities/isaac-gym-isaac-lab.md`](../../wiki/entities/isaac-gym-isaac-lab.md)

## 当前提炼状态

- [x] 摘要与主方法摘录
- [x] wiki 页面映射确认
