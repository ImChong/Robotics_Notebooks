---
type: entity
tags: ['paper', 'vla', 'reinforcement-learning', 'flow-matching', 'online-rl']
status: complete
updated: 2026-08-23
arxiv: "2608.15139"
related:
  - ../methods/vla.md
  - ../methods/reinforcement-learning.md
  - ../formalizations/probability-flow.md
  - ../overview/vla-robustness-9-papers-technology-map.md
sources:
  - ../../sources/papers/structrl_arxiv_2608_15139.md
  - ../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-08-23.md
  - ../../sources/sites/structrl.md
summary: "StructRL（arXiv:2608.15139）：动作空间结构化探索缓解 flow-VLA 在线 RL 的 Structured Noise Dilution；项目页无 GitHub。"
---

# StructRL

**StructRL: Structured Action-Space Exploration for Flow-Based VLAs**（[arXiv:2608.15139](https://arxiv.org/abs/2608.15139)，[项目页](https://flyfaerss.github.io/structrl/)）——复旦大学；上海人工智能实验室等（见 arXiv 作者列表）。

## 一句话定义

**VLA 在线适配的探索噪声必须作用在真正执行的动作上。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作大模型 |
| ODE | Ordinary Differential Equation | 确定性 flow 解码 |
| OOD | Out-of-Distribution | 分布外泛化评测 |

## 为什么重要

- 纳入 [具身智能小站 2026-08-23 九篇盘点](../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-08-23.md) 的「长视野 VLA → 动作分布 → 真实交互」主线。
- 开源状态（入库日）：**待发布**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 复旦大学；上海人工智能实验室等（见 arXiv 作者列表） |
| **出处** | arXiv:2608.15139（2026-08） |
| **开源** | **待发布** |

### 流程总览

```mermaid
flowchart LR
  obs[观测/指令] --> core[核心方法模块]
  core --> act[动作/规划输出]
  act --> rob[仿真或真机闭环]
```

## 评测

| 项 | 内容 |
|----|------|
| **主结果** | LIBERO / ManiSkill 上三类 flow-VLA backbone；真机 Pick Banana **4%→84%**、Plug Charger **0%→100%**（10 条 SFT + StructRL+AWAC）。 |

- 数据出处：[ingest 摘录](../../sources/papers/structrl_arxiv_2608_15139.md)。

## 结论

**链内去噪会稀释结构化探索——应把随机性绑定到最终执行动作。**

- 识别 Structured Noise Dilution 现象
- 确定性 ODE decoder + 动作空间 AR(1) 噪声
- position/rotation/gripper 分组尺度
- last-step replay 提供可训练信号
- 项目页截至入库日无公开代码链

## 源码运行时序图

**不适用**（截至 **2026-08-23**）：项目页未列可运行代码仓库。

## 与其他页面的关系

- [vla](../methods/vla.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [probability-flow](../formalizations/probability-flow.md)
- [vla-robustness-9-papers-technology-map](../overview/vla-robustness-9-papers-technology-map.md)

## 参考来源

- [structrl_arxiv_2608_15139](../../sources/papers/structrl_arxiv_2608_15139.md)
- [structrl](../../sources/sites/structrl.md)
- [wechat_embodied_station_9_papers_open_source_2026-08-23](../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-08-23.md)

## 推荐继续阅读

- [arXiv:2608.15139](https://arxiv.org/abs/2608.15139)
- [项目页](https://flyfaerss.github.io/structrl/)
