---
type: entity
tags:
  - paper
  - mpc
  - reinforcement-learning
  - navigation
  - multi-robot
  - control-barrier-function
  - hku
  - sustech
status: complete
updated: 2026-08-25
arxiv: "2608.21175"
code: https://github.com/hanruihua/srl_mpc
related:
  - ../methods/model-predictive-control.md
  - ../methods/reinforcement-learning.md
  - ../concepts/mpc-wbc-integration.md
  - ../overview/open-source-8-papers-technology-map.md
sources:
  - ../../sources/papers/srl_mpc_arxiv_2608_21175.md
  - ../../sources/sites/srl-mpc-hanruihua.md
  - ../../sources/repos/srl-mpc.md
  - ../../sources/blogs/wechat_embodied_station_8_papers_open_source_2026-08-25.md
summary: "SRL-MPC（arXiv:2608.21175，HKU）：RL 读取 GSF 在线调 MPC 参数 + 形状感知 HOCBF；25 机器人密集场景 86.7% 成功率；代码待发布。"
---

# SRL-MPC：形状感知强化学习 MPC

**SRL-MPC: Shape-Aware Reinforcement Learned Model Predictive Control**（[arXiv:2608.21175](https://arxiv.org/abs/2608.21175)，[项目页](https://hanruihua.github.io/srl_mpc_project/)）由 **香港大学（HKU）** 等提出：在 **不简化几何形状** 的前提下，用 **几何分离特征（GSF）** 构造 **高阶控制屏障函数（HOCBF）** 约束，并由 **RL 策略** 实时更新 MPC 跟踪权重与安全距离，在异构密集机器人群中导航。

## 一句话定义

**让学习负责调参与适应、让优化器负责显式形状安全——RL 不替代 MPC，而是给形状感知 HOCBF-MPC 装上在线增益调度。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域显式优化规划 |
| HOCBF | High-Order Control Barrier Function | 高阶离散 CBF 安全约束 |
| GSF | Geometric Separation Feature | 支撑函数导出的分离方向与最小距离 |
| RL | Reinforcement Learning | 读取邻域 GSF 输出 MPC 参数更新 |
| OOD | Out-of-Distribution | 训练外形状/密度/动力学配置 |

## 为什么重要

- **密集异构人群：** 传统 VO/RL 常假设同质圆盘或稀疏场景，密度升高时碰撞或停滞。
- **可解释混合控制：** 相对端到端 RL，执行动作仍来自显式 MPC，安全结构可审计。
- **形状泛化：** 凸多边形与非凸并集（凸分量并）无需手工重调权重。
- **强基线差距：** 25 机器人设置比 SARL（21%）高 **+65.7 pp**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 香港大学（HKU）；南方科技大学（SUSTech）；密歇根大学；深圳先进院等 |
| **训练分布** | 15 机器人随机凸多边形人群 |
| **开源** | **待发布** — `hanruihua/srl_mpc` README 写明录用后释码 |

### 流程总览

```mermaid
flowchart LR
  GSF[邻域 GSF 特征] --> RL[RL 策略\n更新 MPC 权重/安全距]
  RL --> MPC[形状感知 HOCBF-MPC]
  MPC --> CTRL[差速/全向底盘控制]
  GEOM[支撑函数几何] --> GSF
```

## 工程实践

| 项 | 建议 |
|----|------|
| **部署读法** | 把 RL 输出当作 MPC 成本与安全裕度的调度器，而非直接动作 |
| **形状建模** | 机体 footprint 用凸集表示；非凸体拆分为凸并集 |
| **密度扩展** | 15 机器人训练策略可直接评测 20/25 机器人密集场景（论文报告 91.0%/86.7%） |
| **复现** | 截至入库日跟项目页演示与论文；源码待官方释出 |

## 局限与风险

- 交叉几何 OOD 族成功率降至 ~59%，说明异构形状混部仍难。
- 源码未发布，工程复现暂依赖论文与演示。
- 真机对比为 preliminary demo，需更多公开硬件细节。

## 评测

| 设置 | 结果 |
|------|------|
| 20 机器人密集 | **91.0% ± 2.6 pp**（三策略平均） |
| 25 机器人密集 | **86.7% ± 0.6 pp** |
| vs SARL @ 25 | **+65.7 pp** |
| 非凸并集 OOD | **96.7% ± 0.6%** |
| 异构动力学 OOD | **95.0% ± 1.7%** |

## 结论

**形状感知安全约束 + RL 调参，是在密集异构人群中保留 MPC 可解释性的务实路线。**

- GSF 固定维表征避免边数爆炸的锥约束
- HOCBF _residual 软惩罚嵌入局部 MPC 子问题
- RL 仅更新权重与安全距离，执行仍走显式优化
- 密度升高时相对 VO/RL 基线优势扩大
- 官方代码待录用后发布，选型先以项目页评测为准

## 源码运行时序图

| 项 | 说明 |
|----|------|
| **源码运行时序图** | **不适用**（截至 **2026-08-25** 源码待发布；仅有项目页仿真/真机演示） |

## 与其他页面的关系

- [Model Predictive Control](../methods/model-predictive-control.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [mpc-wbc-integration](../concepts/mpc-wbc-integration.md)
- [open-source-8-papers-technology-map](../overview/open-source-8-papers-technology-map.md)

## 参考来源

- [srl_mpc_arxiv_2608_21175](../../sources/papers/srl_mpc_arxiv_2608_21175.md)
- [srl-mpc-hanruihua](../../sources/sites/srl-mpc-hanruihua.md)
- [srl-mpc](../../sources/repos/srl-mpc.md)
- [wechat_embodied_station_8_papers_open_source_2026-08-25](../../sources/blogs/wechat_embodied_station_8_papers_open_source_2026-08-25.md)

## 推荐继续阅读

- [arXiv:2608.21175](https://arxiv.org/abs/2608.21175)
- [SRL-MPC 项目页](https://hanruihua.github.io/srl_mpc_project/)
