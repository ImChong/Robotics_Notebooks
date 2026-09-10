---
type: entity
tags: [paper, vlm, manipulation, discrete-actions, franka]
status: complete
updated: 2026-09-10
arxiv: "2609.10522"
code: https://github.com/showlab/Show-Harness
related:
  - ../methods/imitation-learning.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/vlm-manipulation-11-papers-technology-map.md
sources:
  - ../../sources/papers/show-harness_arxiv_2609_10522.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md
summary: "离散语义微动作单元连接 VLM 与机器人执行；Franka/AgileX 164 真机 episode；showlab/Show-Harness 已开源。"
---

# Show-Harness（arXiv:2609.10522）

**Show-Harness**（[Show-Harness: Just a VLM Agent Can Play Robots](https://arxiv.org/abs/2609.10522)）来自 [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)。离散语义微动作单元连接 VLM 与机器人执行；Franka/AgileX 164 真机 episode；showlab/Show-Harness 已开源。

## 一句话定义

**零样本 frontier VLM 与 2B 微调 VLM 在跨任务/环境/本体上优于对照（平行夹爪单臂/双臂）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 从专家示范学习策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| RL | Reinforcement Learning | 强化学习 |
| CEM | Cross-Entropy Method | 采样优化动作/轨迹的规划器 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- 纳入本期 **VLM 控制 / 世界模型 / 灵巧操作 / 规划 / 评测** 主线之一。
- 开源状态：**已开源**（步骤 2.5 核查，2026-09-10）。
- 与 [11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.10522](https://arxiv.org/abs/2609.10522) |
| **项目页** | https://showlab.github.io/Show-Harness |
| **代码** | https://github.com/showlab/Show-Harness |
| **开源** | **已开源** |
| **文内指标** | 零样本 frontier VLM 与 2B 微调 VLM 在跨任务/环境/本体上优于对照（平行夹爪单臂/双臂）。 |


## 源码运行时序图

```mermaid
sequenceDiagram
  participant U as 用户/脚本
  participant R as 官方仓库入口
  participant M as 模型/规划器
  participant E as 仿真或真机环境
  U->>R: clone + 安装依赖
  U->>M: 加载配置/权重
  U->>E: rollout / 规划 / 控制
  M-->>E: 动作或轨迹
  E-->>U: 成功率/指标日志
```


## 结论

**Show-Harness 值得按「已开源」边界阅读：先核对仓库是否可跑，再引用文内成功率数字。**

1. 索引来源为公众号导读，实验细节以 arXiv PDF 为准。
2. 开源结论：**已开源**。
3. 选型时对照 [11 篇地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中相邻节点，避免重复造页。

## 关联页面

- [VLM 与操作 11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md)
- [模仿学习 (Imitation Learning)](../methods/imitation-learning.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [show-harness_arxiv_2609_10522.md](../../sources/papers/show-harness_arxiv_2609_10522.md)
- [wechat 11篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)
- [arXiv:2609.10522](https://arxiv.org/abs/2609.10522)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.10522)
- [项目页](https://showlab.github.io/Show-Harness)
- [GitHub](https://github.com/showlab/Show-Harness)
