---
type: entity
tags: [paper, curated-index, awesome-sim2real, longchao-sim2real]
status: complete
updated: 2026-10-06
arxiv: "1612.03801"
summary: "LongchaoDa AwesomeSim2Real 收录；分组 Other Environments。 本页是清单索引，细节以原文为准。"
related:
  - ../entities/awesome-sim2real.md
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/lc_awesome_sim2real_1612_03801_deepmind-lab.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/repos/awesome-sim2real.md
---

# DeepMind Lab

**DeepMind Lab** 收录于 [AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real) **第 068/139** 篇，分组 **Other Environments**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LongchaoDa AwesomeSim2Real 收录；分组 Other Environments。 本页是清单索引，细节以原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型/基础模型增强迁移 |

## 为什么重要

- LongchaoDa AwesomeSim2Real 收录；分组 Other Environments。 本页是清单索引，细节以原文为准。
- 想横向对照同一分组的其他工作，可以从 [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) 逐条展开。
- 顺着列表实体 [AwesomeSim2Real](../entities/awesome-sim2real.md) 与站内 Sim2Real 方法/任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    ENG["游戏引擎: ioquake3 版 Quake III Arena<br/>自定义资产, 原引擎 C 代码仅在必要处修改"] --> API["强化学习 API: C API + Python 绑定<br/>锁步 lock-step: 每次推进一个或多个模拟步, 之后暂停等待下一个动作"]
    API -->|"观测: 标量奖励 / RGB 或 RGBD 像素 / 平动与角速度 可选"| AG["智能体 或 人类"]
    AG -->|"动作: 前后移动, 左右平移, 蹲, 跳, 上下左右看, 激光标记"| API
    LVL["关卡: Quake III 关卡 .pk3 文件 由 .map 经 q3map2 / bspc 工具生成<br/>或 Text Levels 文本关卡"] --> ENG
    LUA["Lua 关卡 API<br/>bot 逻辑 / 道具拾取 / 自定义观测 / 重启 / 奖励方案"] -.->|"进一步定制每个关卡"| LVL
    LVL -.-> EX["示例关卡四类<br/>水果采集 / 静态地图迷宫导航 / 程序生成迷宫 / 激光标记"]
```

图依据：平台架构与交互循环图，论文 DeepMind Lab Research Platform 一节（Observations、Actions、Example levels、Technical Details、API for agents and humans、Level generation）与 Fig. 1、Fig. 2、Fig. 4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 068/139 |
| 分组 | Other Environments |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/1612.03801> |

## 核心机制（归纳）

### 策展导读要点

LongchaoDa AwesomeSim2Real 收录；分组 Other Environments。 本页是清单索引，细节以原文为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/1612.03801) 为准。
- 横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/lc-awesome-sim2real-technology-map.md) 的 **Other Environments** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome 列表可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DeepMind Lab」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Other Environments**；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的分组标签当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[AwesomeSim2Real](../entities/awesome-sim2real.md)
- 技术地图：[AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- 方法/任务：[sim2real.md](../concepts/sim2real.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/lc_awesome_sim2real_1612_03801_deepmind-lab.md`](../../sources/papers/lc_awesome_sim2real_1612_03801_deepmind-lab.md) — 本条目策展摘录
- [`sources/papers/lc_awesome_sim2real_catalog.md`](../../sources/papers/lc_awesome_sim2real_catalog.md) — 列表总表
- [`sources/repos/awesome-sim2real.md`](../../sources/repos/awesome-sim2real.md)
- 论文：<https://arxiv.org/abs/1612.03801>
- [arXiv PDF](https://arxiv.org/pdf/1612.03801)：已通读全文正文（Introduction、DeepMind Lab Research Platform 各小节、Results and Performance、Conclusion）；帧率表与附图 7-8 的示例截图略读

## 推荐继续阅读

- [AwesomeSim2Real 仓库](https://github.com/LongchaoDa/AwesomeSim2Real)
- [原文](https://arxiv.org/abs/1612.03801)
