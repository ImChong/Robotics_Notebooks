# Flow-Matched Motion Priors（arXiv:2609.15631）

> 来源归档（ingest）

- **标题：** Flow-Matched Motion Priors: Online Optimal-Transport Rewards for Imitation Learning
- **简称：** FMP
- **类型：** paper / motion-prior / imitation-learning / humanoid
- **arXiv：** <https://arxiv.org/abs/2609.15631>
- **PDF：** <https://arxiv.org/pdf/2609.15631>
- **机构：** 清华大学航天工程学院的蒋方华组；中国科学院软件研究所（Chenghua Liu）
- **入库日期：** 2026-09-16
- **一句话说明：** 用熵正则 OT 耦合 rollout 历史与专家动作库，在线 flow matching 学标量运动先验奖励；G1 上优于 AMP 与质心 OT 奖励。

## 开源状态（步骤 2.5，2026-09-16）

**结论：截至入库日 arXiv 与论文 HTML 未列 GitHub / 项目页。**

## 核心摘录

### 摘录 1：问题与动机

- 运动先验需要把策略从当前行为引向示范动作；**AMP** 用判别器给标量奖励，但当策略与专家支撑集相距较远时对抗目标会变得不 informative。
- 朴素 **OT** 把匹配到的专家后继状态做 **barycentric 平均**；跨步态相位平均会削弱关节运动目标。

**对 wiki 的映射：** [paper-fmp-motion-priors](../../wiki/entities/paper-fmp-motion-priors.md)

### 摘录 2：FMP 方法

- **FMP（Flow-Matched Motion Priors）**：在 **rollout 历史 → 专家动作库** 路径上，用 **熵正则 OT** 做耦合；每次策略更新前，用 **flow matching（FM）** 训练神经势能，辅以 endpoint-gradient 监督与 relative-value 校准。
- Actor 仍只吃物理观测，奖励保持 **标量**（与 AMP 接口一致）。

**对 wiki 的映射：** 同上

### 摘录 3：G1 实验（50M transition 对齐）

- 平台：**Unitree G1**，**Isaac Lab**；对照 **AMP**、barycentric OT 奖励与嵌套消融。
- **示范重置**：稳定前向行走 **0.727 m/s**。
- **固定默认姿态初始化**：**0.338 m/s**；跌倒 **129** 次 vs endpoint-only 控制 **243** 次。
- 相对静态 score-gradient teacher，动态 FM 在插值分数 0.25/0.50 处降低 score-increment error，离线拟合时间少 **29%**。

**对 wiki 的映射：** 同上（评测与结论）

## 当前提炼状态

- [x] arXiv HTML 摘要与实验要点核查（2026-09-16）
- [x] wiki 映射：`wiki/entities/paper-fmp-motion-priors.md`
