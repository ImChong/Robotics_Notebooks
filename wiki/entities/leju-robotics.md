---
type: entity
tags: [humanoid, hardware, open-source, leju, kuavo, dataset-ecosystem]
status: complete
updated: 2026-07-17
related:
  - ./humanoid-robot.md
  - ./openlet.md
  - ./lerobot.md
  - ../tasks/manipulation.md
  - ../tasks/loco-manipulation.md
  - ./paper-motion-cerebellum-guidewalk.md
  - ./unitree.md
sources:
  - ../../sources/sites/lejurobot.md
  - ../../sources/sites/openlet-openatom.md
summary: "乐聚机器人：KUAVO（夸父）全尺寸人形与轮臂双形态产品线，覆盖科研开箱、商服导览、工业分拣与训练场数据生态；牵头运营 OpenLET 开源真机数据集社区。"
---

# 乐聚机器人（Leju Robotics）

**乐聚机器人**（[lejurobot.com](https://www.lejurobot.com/zh)）是国内 **全尺寸人形机器人产业化** 代表厂商之一：以 **KUAVO（夸父）** 系列人形与 **OpenLET** 数据社区，把「能买到的整机 + 能下载的真机小时 + LeRobot 复现教程」串成一条 **科研—工业—数据** 闭环。

## 一句话定义

**全尺寸人形硬件平台商 + 真机数据生态运营方：KUAVO 负责机体与场景落地，OpenLET 负责把采集数据与挑战赛资源开源给 VLA/IL 社区。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| KUAVO | — | 乐聚通用人形机器人产品系列（含 4 Pro、5、5-W 等） |
| VR | Virtual Reality | OpenLET 采集使用的沉浸式遥操界面 |
| IL | Imitation Learning | 模仿学习；OpenLET 提供 Kuavo + LeRobot 范例 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略；LET 数据可作后训练语料 |
| CRAIC | China Robot and AI Competition | 社区栏目之一的人形机器人挑战赛 |
| REAL-I | — | ICRA 背书的具身智能工业任务挑战赛系列 |

## 为什么重要

- **全尺寸 + 开箱科研叙事：** 与教育级小型人形不同，KUAVO 强调 **科研平台开箱即用** 与 **工业/商服场景方案**，适合作为 **loco-manipulation 真机** 对照 [Unitree](./unitree.md) 的国产全尺寸选项。
- **数据不只是宣传：** [OpenLET](./openlet.md) 在开放原子社区发布 **>60,000 分钟** LET 系列真机数据，并配套 **AtomGit 仓库 + LeRobot 转换/部署**——降低「有整机但缺可训练数据」的摩擦。
- **学术—产业交叉：** 与哈工大等联合的人形运动控制论文（如 [GuideWalk](./paper-motion-cerebellum-guidewalk.md)）说明其在 **运动控制研究圈** 亦有持续曝光。

## 核心产品/方案（策展）

| 层级 | 代表 | 要点 |
|------|------|------|
| **硬件** | KUAVO 4 Pro、KUAVO 5 / 5-W | 全尺寸人形；5 系强调 **轮臂双形态** 与工业场景升级 |
| **公开案例** | 夸父火炬手、亚冬会传递等 | 运动控制与大活动部署的品牌展示（非学术 benchmark） |
| **场景包** | 科研 / 商服 / 工业 / 训练场 | 从实验室到展厅导览、汽车 3C 分拣、数据训练场 |
| **数据生态** | [OpenLET](./openlet.md) | LET 全身运控、灵巧手、轮臂基础操作 + 挑战赛 |

## 开源与复现入口

| 类型 | 入口 | 说明 |
|------|------|------|
| 数据集社区 | [openlet.openatom.tech](https://openlet.openatom.tech/) | 社区首页、专栏与仓库聚合 |
| 旗舰数据仓 | [LET-Base-Dataset](https://atomgit.com/OpenLET/LET-Base-Dataset) | 轮臂基础操作（Kuavo 4 Pro / 5W） |
| IL 工具链 | [kuavo-manip-open](https://atomgit.com/OpenLET/kuavo-manip-open) | rosbag→parquet、训练、仿真与真机部署 |
| 整机官网 | [lejurobot.com/zh](https://www.lejurobot.com/zh) | 产品参数与方案咨询（**非** 整机 CAD 全开源） |

> **开源边界：** OpenLET 侧 **数据与教程已开源**；KUAVO 整机为 **商业产品**，不等同于 [灵犀 X1](./agibot-lingxi-x1.md) 类「硬件图纸全开放」路线。

## 常见误区或局限

- **误区：乐聚 = 只有整机没有数据。** 运营方身份包含 **OpenLET 数据枢纽**；研究选型应同时看 [OpenLET](./openlet.md) 子集与采集协议。
- **误区：LET 数据可零样本跨到任意人形。** 数据来自 **Kuavo  kinematics / 传感器布局**；迁移到 G1/H1 等仍需重定向或再采集。
- **局限：** 相对 Unitree，国际论文 benchmark 中的 **出现频率仍较低**；社区影响力依赖 OpenLET 持续更新与赛事牵引。

## 关联页面

- [OpenLET](./openlet.md) — LET 数据集与 AtomGit 仓库
- [LeRobot](./lerobot.md) — `kuavo-manip-open` 数据格式与训练栈
- [人形机器人](./humanoid-robot.md) — 硬件品类总览
- [Manipulation](../tasks/manipulation.md) — LET 基础操作任务语境
- [Unitree](./unitree.md) — 另一全尺寸/人形硬件主线对照

## 推荐继续阅读

- OpenLET 社区：<https://openlet.openatom.tech/>
- LET-Base-Dataset：<https://atomgit.com/OpenLET/LET-Base-Dataset>
- 乐聚官网：<https://www.lejurobot.com/zh>

## 参考来源

- [lejurobot.md](../../sources/sites/lejurobot.md) — 官网产品/方案策展
- [openlet-openatom.md](../../sources/sites/openlet-openatom.md) — 数据社区与仓库索引
