# SRL-MPC（arXiv:2608.21175）

> 来源归档（ingest）

- **标题：** SRL-MPC: Shape-Aware Reinforcement Learned Model Predictive Control
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.21175>
  - <https://hanruihua.github.io/srl_mpc_project/>
- **代码：** <https://github.com/hanruihua/srl_mpc>（README：**录用后释码**）
- **机构：** 香港大学（HKU）；南方科技大学（SUSTech）；密歇根大学；深圳先进院等
- **入库日期：** 2026-08-25
- **一句话说明：** 用支撑函数几何分离特征构造形状感知 HOCBF，RL 策略读取邻域 GSF 实时更新 MPC 参数；密集 25 机器人场景成功率 **86.7%**，显著优于 SARL 等基线。

## 核心摘录（MVP）

### 1) 形状感知 HOCBF + GSF

- **摘录要点：** 基于支撑函数变换得到固定维几何分离特征（GSF），构造二阶离散 HOCBF 约束；凸多边形与非凸并集均可表示，无需把异构机体简化为同质圆盘。
- **对 wiki 的映射：**
  - [SRL-MPC](../../wiki/entities/paper-srl-mpc.md) — 方法核心。
  - [Model Predictive Control](../../wiki/methods/model-predictive-control.md) — 显式优化执行层。

### 2) RL 调参、MPC 执行

- **摘录要点：** RL 不替代规划器，而是自适应路径跟踪权重、控制努力权重与安全距离；执行控制仍由显式 MPC 求解，保留可解释安全结构。
- **对 wiki 的映射：**
  - [SRL-MPC](../../wiki/entities/paper-srl-mpc.md) — 混合控制定位。
  - [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) — 学习侧角色。

### 3) 密集场景与 OOD 评测

- **摘录要点：** 15 机器人训练分布上独立训练三策略；20/25 机器人密集场景平均成功率 **91.0% / 86.7%**；25 机器人设置比最强外部基线 SARL（21%）高 **+65.7 pp**；含非凸并集、异构动力学等 OOD 族。
- **对 wiki 的映射：**
  - [SRL-MPC](../../wiki/entities/paper-srl-mpc.md) — 评测表。

### 4) 开源状态（截至 2026-08-25）

- **摘录要点：** GitHub 仓 `hanruihua/srl_mpc` 已建，README 写明 **「The source code will be released upon acceptance of the paper」** → **待发布**。
- **对 wiki 的映射：**
  - [srl-mpc 仓库归档](../repos/srl-mpc.md)
  - [SRL-MPC 项目页](../sites/srl-mpc-hanruihua.md)

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-srl-mpc.md` 新建
