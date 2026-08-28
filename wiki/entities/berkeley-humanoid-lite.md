---
type: entity
tags: [paper, humanoid, hardware, open-source, berkeley, reinforcement-learning, qdd, humanoid-paper-notebooks, cycloidal, actuator]
status: complete
updated: 2026-07-25
arxiv: "2504.17249"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ./odri-solo-and-bolt.md
  - ./internal-cycloidal-actuator.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
  - ../methods/reinforcement-learning.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
  - ../../sources/papers/humanoid_pnb_berkeley-humanoid-lite-an-open-source-accessible.md
  - ../../sources/repos/berkeley_humanoid_lite.md
  - ../../sources/sites/berkeley_humanoid_lite.md
  - ../../sources/personal/open_source_qdd_actuator_learning_curator.md
summary: "Berkeley Humanoid Lite（BHL）：UC Berkeley 低成本准直驱开源人形；~15:1 3D 打印摆线关节、电机/控制参数、BOM、底层控制、Isaac Lab 训练与实机部署——学「执行器如何装进人形」的优先整机参考。"
---

# Berkeley Humanoid Lite（BHL）

## 一句话定义

**Berkeley Humanoid Lite** 是 **UC Berkeley Hybrid Robotics** 的 **轻量人形** 开源方案：门户 **[lite.berkeley-humanoid.org](https://lite.berkeley-humanoid.org/)**，源码与仿真/学习脚本在 **[Berkeley-Humanoid-Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)**（MIT）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| QDD | Quasi-Direct Drive | 准直驱，低减速比、高背驱动性的作动方案 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |

## 为什么重要

- **低成本 QDD 叙事**：在 [开源人形硬件对比](./open-source-humanoid-hardware.md) 中常与商业整机对照，作为 **动力学透明 + RL 友好** 的参考轴。
- **最接近「人形关节需求」的开源整机**：含整机 CAD、3D 打印摆线、电机与关节参数、控制器配置、BOM、底层控制、**Isaac Lab** 训练与实机部署。
- 在 [开源 QDD 学习路线](../comparisons/open-source-qdd-actuator-projects.md) 中排在 ODRI 之后：学完力控关节后，看执行器如何进人形。

## 关节与电机要点（策展）

| 项 | 内容 |
|----|------|
| 减速 | 约 **15:1** 3D 打印摆线 + 外转子电机 |
| 电机 | 约 **14 对极**；力矩常数约 **0.1176 N·m/A** |
| 控制 | 公开电流环、位置环、速度环等参数 |
| 值得对照 | 髋/膝布置；电机参数→关节参数映射；电流/力矩限制 |

## 论文与深读状态

| 字段 | 内容 |
|------|------|
| 论文 | *Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot*，<https://arxiv.org/abs/2504.17249> |
| Paper Notebooks 分类 | 12_Hardware_Design（[分类父节点](../overview/paper-notebook-category-12-hardware-design.md)） |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)）；笔记完成后在本页链向笔记站并深化归纳 |

## 开源入口

| 类型 | 链接 | 状态 |
|------|------|------|
| 项目门户 | [lite.berkeley-humanoid.org](https://lite.berkeley-humanoid.org/) | 已发布 |
| GitBook 文档 | [berkeley-humanoid-lite.gitbook.io/docs](https://berkeley-humanoid-lite.gitbook.io/docs) | 已发布 |
| 主仓库 | [HybridRobotics/Berkeley-Humanoid-Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite) | **已开源**（MIT） |

## 局限与风险

- 官方已指出：当前 **3D 打印摆线** 在高性能运动中偏脆弱，后续版本可能改为成品关节。
- **适合学习与验证，不适合原样用于重型人形**。
- 可与 [Internal Cycloidal](./internal-cycloidal-actuator.md)（电机—减速一体）及 [Urs 等打印 QDD 论文](./paper-3d-printed-open-source-actuators-legged.md)（系统热/寿命评测）对照。

## 关联页面

- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)
- [ODRI Solo / Bolt](./odri-solo-and-bolt.md)
- [开源人形硬件方案对比](./open-source-humanoid-hardware.md)
- [Paper Notebooks 分类父节点：12 Hardware Design](../overview/paper-notebook-category-12-hardware-design.md)
- [强化学习](../methods/reinforcement-learning.md)
- [力矩电机设计纵深](../../roadmap/depth-torque-motor-design.md)

## 推荐继续阅读

- [开源人形硬件方案对比 · Berkeley Humanoid 段](./open-source-humanoid-hardware.md#1-berkeley-humanoid-准直接驱动派)
- 主仓 README 与 GitBook 关节/控制参数章节

## 参考来源

- [berkeley_humanoid_lite.md](../../sources/repos/berkeley_humanoid_lite.md)
- [berkeley_humanoid_lite 项目页](../../sources/sites/berkeley_humanoid_lite.md)
- [开源 QDD 执行器学习策展](../../sources/personal/open_source_qdd_actuator_learning_curator.md)
- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
- [humanoid_pnb_berkeley-humanoid-lite-an-open-source-accessible.md](../../sources/papers/humanoid_pnb_berkeley-humanoid-lite-an-open-source-accessible.md)
- 论文：<https://arxiv.org/abs/2504.17249>
