---
type: overview
updated: 2026-05-07
summary: "按人形、四足与腿足梳理当前知名度高、常被产业与媒体报道的商业机器人平台，并指向本库已有实体页与深度对比页。"
sources:
  - ../../sources/repos/notable-commercial-robot-platforms.md
related:
  - ../entities/humanoid-robot.md
  - ../entities/unitree.md
  - ../entities/boston-dynamics.md
  - ../entities/anymal.md
  - ../queries/hardware-comparison.md
---

# 市面知名机器人平台纵览

## 一句话定义

本页回答：**除了少数明星项目外，产业与新闻里还经常出现哪些人形、四足与腿足平台**，它们大致属于哪条技术–商业路线，以及在本知识库里应去哪里深挖。

## 为什么重要

做选型、写综述或对齐仿真模型时，如果只盯着一两家公司，容易误判「行业平均水平」。把 **高频出镜的平台族谱** 摊开，可以快速判断：

- 你需要对标的是 **科研易获得硬件**（如宇树）、**工业落地闭环**（如 Agility / Apptronik）、还是 **全栈 AI 叙事**（如 Figure / Sanctuary）。
- 四足与人形的 **供应商重叠**（例如同一家公司既做四足又做人形）会影响采购、售后与仿真资产迁移策略。

## 人形：常见「阵营」速览

下列品牌在英文科技媒体、国内产业报道与展会中曝光度较高；**不等同于性能排名**，亦不构成采购建议。

### 北美 / 全球化总部（全能与物流叙事）

| 品牌（例子） | 典型标签 | 本库延伸阅读 |
|--------------|-----------|----------------|
| Boston Dynamics | Atlas / Spot；液压转型全电 Atlas | [Boston Dynamics](../entities/boston-dynamics.md) |
| Agility Robotics | Digit；仓储物流人形 | （暂无独立实体页，见下表外链） |
| Figure AI | Figure 02、Helix VLA | [Figure AI](../entities/figure-ai.md) |
| 1X Technologies | EVE、NEO | [1X Technologies](../entities/1x-technologies.md) |
| Tesla | Optimus；汽车供应链叙事 | [人形机器人总览表](../entities/humanoid-robot.md) |
| Apptronik | Apollo；工业通用人形 | 同上速览表 |
| Sanctuary AI | Phoenix、Carbon | 见 [原始资料链接索引](../../sources/repos/notable-commercial-robot-platforms.md) |

### 中国及亚太（出货与场景绑定深）

| 品牌（例子） | 典型标签 | 本库延伸阅读 |
|--------------|-----------|----------------|
| Unitree | H1、G1；Go2、B2 | [Unitree](../entities/unitree.md)、[G1](../entities/unitree-g1.md) |
| Fourier Intelligence | GR-1 等全身人形 | [硬件对比](../queries/hardware-comparison.md) |
| UBTECH | Walker 系列 | 同上 |
| LimX Dynamics、Booster Robotics 等 | 融资与赛事曝光高 | [Booster RoboCup Demo](../entities/booster-robocup-demo.md)（Booster 相关） |

更多官网入口见 [原始资料：平台链接索引](../../sources/repos/notable-commercial-robot-platforms.md)。

## 四足与腿足：常见平台

| 类型 | 代表厂商 / 产品 | 说明 |
|------|-------------------|------|
| 工业四足 | Boston Dynamics **Spot** | 巡检、测绘；与 [Boston Dynamics](../entities/boston-dynamics.md) 同一技术谱系 |
| 高端科研 / 工业四足 | ANYbotics **ANYmal** | ETH 衍生，防爆与工业场景强；见 [ANYmal](../entities/anymal.md) |
| 科研与消费级四足主流量产 | Unitree **Go2 / B2** 等 | 生态与价格带宽大；见 [Unitree](../entities/unitree.md) |
| 国内行业四足 | DEEP Robotics **绝影** 系列 | 电力、应急、测绘等场景常见 |
| 开发者生态 | 小米 **CyberDog** 等 | 偏品牌与开发者社区，迭代快 |

## 常见误区

- **曝光量 ≠ 可买性 ≠ 可复现性**：概念车、实验室样机与稳定供货三者差别很大。
- **「全栈自研 AI」与「运控扎实」不总在一个团队**：评估时要拆开看硬件、中间件、策略层与数据管线。
- **国内四足 ≠ 只有一家**：除宇树外，云深处、小米等在垂直场景同样高频出现，选型要看场景认证与售后而不是只看参数表。

## 关联页面

- [人形机器人](../entities/humanoid-robot.md)
- [主流人形机器人硬件对比](../queries/hardware-comparison.md)
- [Query：人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Unitree](../entities/unitree.md)
- [Boston Dynamics](../entities/boston-dynamics.md)
- [ANYmal](../entities/anymal.md)

## 参考来源

- [notable-commercial-robot-platforms](../../sources/repos/notable-commercial-robot-platforms.md)

## 推荐继续阅读

- [IEEE Spectrum Robotics](https://spectrum.ieee.org/topic/robotics/) — 产业动态与产品发布跟踪
- [Agility Robotics（Digit）](https://www.agilityrobotics.com/) — 物流人形商业部署案例入口
