# 最大误区：Sim-to-Real 不是训完之后的事情！从辨识到适应，这些工作贯穿全程

> 来源归档（blog / 微信公众号）

- **标题：** 最大误区：Sim-to-Real 不是训完之后的事情！从辨识到适应，这些工作贯穿全程
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；编译/整理；课程宣传文）
- **原始链接：** https://mp.weixin.qq.com/s/6rbLz_6nQz9z6kma9K4BFQ
- **发表日期：** 2026-07-28（frontmatter）
- **入库日期：** 2026-07-28
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`；`--no-images`）；正文约 1.17 万字 / 27 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_shenlan_sim2real_sysid_to_adaptation_2026-07-28.md](../raw/wechat_shenlan_sim2real_sysid_to_adaptation_2026-07-28.md)
- **课程出处（文末声明）：** 深蓝学院联合纽卡斯尔大学潘为《四足机器人：从动力学建模到强化学习》
- **一句话说明：** 把 Sim2Real 定位为「从系统辨识到部署后持续校准」的闭环工程，而非训练结束后的独立迁移步骤；按误差类型分流到校准 / 域随机化 / 在线适应，并以分层安全独立于策略。

## 核心摘录（归纳，非全文）

### 主判断

- **误区：** 把迁移当成「训完再做」的一步 → 辨识、训练、部署彼此割裂；仿真参数训后不再更新；漂移无反馈；前馈仍用过期标定。
- **正读：** Sim2Real 应从 **SysID 阶段启动**，并在运行中持续校准。
- **Gap 先分解再动手：** 参数误差 / 难建模动态与环境 / 观测误差；忌「一失败就盲目扩大 DR」。

### 误差处理分流（文内工程准则）

| 误差类型 | 优先手段 |
|----------|----------|
| 可建模参数偏差（质量、惯量、关节摩擦等） | **系统辨识 + 前馈补偿** |
| 难完整建模的动态/环境（回差、柔性、温升、接触） | **域随机化（围绕已校准基准）** |
| 随时间变化的工况（地形 μ、负载、电压） | **在线适应（如 RMA）** |
| 部署风险 | **分层安全（独立于策略）** |

### 系统辨识（物理基准）

- 流程：实机激励轨迹 → 同控制信号仿真回放 → 优化摩擦/惯量等使响应对齐。
- 注意：参数并非越多越好（激励不足易过拟合）；目标是为 RL 提供**合理基准**，再让 DR 覆盖公差与测量误差。
- 锚点：2018 Minitaur — 先电机与延迟 SysID，再随机化，跑跳步态迁移真机。

### 训练面向实机约束

- **观测：** 禁止部署不可得的特权信息直入学生策略；ANYmal 式盲行依赖本体感受 + 历史；教师–学生分离特权与机载观测。
- **动作：** 常见「低频策略目标关节位 + 高频 PD」分工；RL 做全局非线性决策，经典反馈做局部快速扰动。
- **奖励：** 除速度跟踪外，加姿态、平滑、力矩/能耗惩罚，避免仿真高分但真机过热/冲击。

### 域随机化与课程

- DR 范围应依据硬件公差与测量误差，过大 → 保守次优。
- Curriculum：先平地/微扰，再地形与外扰；Rudin et al. *Learning to Walk in Minutes*（CoRL 2021 / 文集 2022）游戏启发课程 + 大规模并行。

### 部署：确定性补偿 + 在线适应 + 敏捷扩展

- **摩擦前馈：** 已辨识摩擦进底层按速度补偿，减轻策略负担。
- **RMA：** 适应模块从近期状态–动作历史推断环境隐变量，无需直接测 μ/负载。
- **敏捷：** Robot Parkour — 专家技能蒸馏为第一人称深度统一策略；ANYmal Parkour — 保留技能库 + 高层导航选技。迁移额外覆盖深度噪声、光照与极限技能切换。

### 分层安全

- 物理急停、驱动电流限、机械止挡、软件力矩/软限位、跌倒检测等；算法 / 控制板 / 驱动 / 机械多层，不依赖单一机制。

### 文内闭环总结

SysID 基准 → 观测/奖励对齐实机 → DR + Curriculum 泛化 → 前馈消可建模误差 → 在线适应动态因素 → 分层安全降部署风险。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [Sim2Real 闭环误差分层工程](../../wiki/queries/sim2real-closed-loop-engineering.md) | **主沉淀页**：误区纠正、误差分流、六段闭环 |
| [Sim2Real](../../wiki/concepts/sim2real.md) | 概念总览；补「非训后一步」误区与本文交叉 |
| [System Identification](../../wiki/concepts/system-identification.md) | SysID 基准 + 勿在错误默认上放大 DR |
| [Domain Randomization](../../wiki/concepts/domain-randomization.md) | 围绕校准基准的随机范围 |
| [Curriculum Learning](../../wiki/concepts/curriculum-learning.md) | 难度调度与并行仿真 |
| [Privileged Training](../../wiki/concepts/privileged-training.md) | 教师特权 vs 学生机载观测 |
| [RMA](../../wiki/entities/paper-rma-rapid-motor-adaptation.md) | 在线适应代表 |
| [Sim2Real Checklist](../../wiki/queries/sim2real-checklist.md) / [Gap 缩减](../../wiki/queries/sim2real-gap-reduction.md) | 工程清单与根因工具箱 |
| [Safety Filter](../../wiki/concepts/safety-filter.md) / [Robot Safety FSM](../../wiki/concepts/robot-safety-state-machine.md) | 分层安全侧 |
| [ANYmal Parkour](../../wiki/entities/paper-notebook-anymal-parkour-robust-perceptive-locomotion.md) | 技能库 + 高层选技 |
| [四足敏捷 Sim2Real（RSS 2018）](../../wiki/entities/paper-quadruped-agile-sim2real-rss2018.md) | 早期 SysID/DR 敏捷迁移参照 |

## 开源 / 项目页核查

- 本文为课程宣传向综述编译，**无独立项目页或官方代码仓**。
- 文内锚点论文的开源状态以各 wiki 实体页 / `sources/papers/` 为准（RMA、legged_gym 路线、Parkour 等），不在本 blog 归档中臆断。
