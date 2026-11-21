---
type: overview
tags: [loco-manipulation, humanoid, category-hub, survey]
status: complete
updated: 2026-09-15
original:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
summary: "人形 Loco-Manip 161 篇 · 02 上半身中心控制与移动操作接口（24 篇）— 手臂、躯干、根部和末端执行器之间的协同控制。"
related:
  - ./humanoid-loco-manip-161-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# Loco-Manip 161 分类 02：上半身中心控制与移动操作接口

> **分类页**：**02 上半身中心控制与移动操作接口**；总地图见 [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)。

## 一句话定义

聚焦人形移动操作中 **上身操作目标、躯干协调与移动底座接口** 的 24 篇代表工作，涵盖根坐标/末端跟踪、上下肢解耦与统一全身策略对比。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |

## 为什么重要
手臂、躯干、根部和末端执行器之间的协同控制

## 核心原理

```mermaid
flowchart LR
  SRC["人手/全身示范或感知目标"] --> IFACE["接口抽象: 上身指令 / 重定向 / 全身遥操"]
  IFACE --> POL["全身策略 / 控制执行"]
```

## 本组论文（24 篇）

| # | 工作 | Wiki 实体 |
|---|------|-----------|
| 032 | AMO | [paper-loco-manip-161-032-amo](../entities/paper-loco-manip-161-135-amo.md) |
| 033 | CEER | [paper-loco-manip-161-033-ceer](../entities/paper-motion-cerebellum-ceer.md) |
| 034 | COLA | [paper-loco-manip-161-034-cola](../entities/paper-notebook-learning-human-humanoid-coordination-for-collabo.md) |
| 035 | CoorDex | [paper-loco-manip-161-035-coordex](../entities/paper-coordex-dexterous-humanoid-loco-manipulation.md) |
| 036 | FALCON | [paper-loco-manip-161-036-falcon](../entities/paper-loco-manip-161-109-falcon.md) |
| 037 | HANDOFF | [paper-loco-manip-161-037-handoff](../entities/paper-motion-cerebellum-handoff.md) |
| 038 | HEX | [paper-loco-manip-161-038-hex](../entities/paper-loco-manip-161-038-hex.md) |
| 039 | HMC | [paper-loco-manip-161-039-hmc](../entities/paper-loco-manip-161-039-hmc.md) |
| 040 | HOMIE | [paper-loco-manip-161-040-homie](../entities/paper-loco-manip-161-040-homie.md) |
| 041 | HiWET | [paper-loco-manip-161-041-hiwet](../entities/paper-loco-manip-161-041-hiwet.md) |
| 042 | Hold | [paper-loco-manip-161-042-hold](../entities/paper-loco-manip-161-042-hold.md) |
| 043 | PILOT | [paper-loco-manip-161-043-pilot](../entities/paper-pilot-perceptive-loco-manipulation.md) |
| 044 | RPL | [paper-loco-manip-161-044-rpl](../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md) |
| 045 | SplitAdapter | [paper-loco-manip-161-045-splitadapter](../entities/paper-splitadapter-load-aware-loco-manipulation.md) |
| 046 | Sumo | [paper-loco-manip-161-046-sumo](../methods/sumo.md) |
| 047 | Thor | [paper-loco-manip-161-047-thor](../entities/paper-hrl-stack-42-thor.md) |
| 048 | ULC | [paper-loco-manip-161-048-ulc](../entities/paper-loco-manip-161-048-ulc.md) |
| 049 | VIRAL | [paper-loco-manip-161-049-viral](../entities/paper-viral-humanoid-visual-sim2real.md) |
| 050 | VOFA | [paper-loco-manip-161-050-vofa](../entities/paper-loco-manip-161-050-vofa.md) |
| 051 | WholeBodyVLA | [paper-loco-manip-161-051-wholebodyvla](../entities/paper-hrl-stack-30-wholebodyvla.md) |
| 052 | 价值网络结构很重要 | [paper-loco-manip-161-052-n052](../entities/paper-loco-manip-161-052-n052.md) |
| 053 | 打开仿真到现实世界的人形像素到动作策略传输之门 | [paper-loco-manip-161-053-n053](../entities/paper-doorman-opening-sim2real-door.md) |
| 054 | 用于人形策略学习的对抗性运动和运动模仿 | [paper-loco-manip-161-054-n054](../entities/paper-amp-survey-07-adversarial-locomotion-and-motion-im.md) |
| 055 | 用于跨人形运动的可扩展和通用全身控制 | [paper-loco-manip-161-055-n055](../entities/paper-loco-manip-161-055-n055.md) |

## 工程实践

- **选型判断核心：** 若任务仅需下盘稳站、上身遥操或轻负载抓放，可先用上下半身指令解耦（如 HOMIE/ULC）；若涉及蹲起、大重物抗扰或全身碰撞，应评估统一接口（WholeBodyVLA/SplitAdapter）。

## 常见误区或局限

- 综述目录页仅作主题归类索引；各项具体成功率、传感器型号与开源状态以各子实体页及原始论文 PDF 核验为准。

## 关联页面

- [人形 Loco-Manip 161 篇技术地图](./humanoid-loco-manip-161-papers-technology-map.md)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)

## 参考来源

- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)

## 推荐继续阅读

- [运动小脑 64 篇技术地图](./humanoid-motion-cerebellum-technology-map.md)
