---
type: entity
tags: [paper, continuum-robots, motion-planning, diffusion]
status: complete
updated: 2026-09-10
arxiv: "2609.09745"
code: https://github.com/qiuke-qiuke/pcc_diffuser
related:
  - ../methods/imitation-learning.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/vlm-manipulation-11-papers-technology-map.md
sources:
  - ../../sources/papers/pccdiffuser_arxiv_2609_09745.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md
summary: "条件扩散生成连续体机器人多模态配置空间路径；障碍物图网络+解析微分运动学；混合测试集 91% 成功率。"
---

# PccDiffuser（arXiv:2609.09745）

**PccDiffuser**（[PccDiffuser: Multi-solution Motion Planning for Continuum Robots](https://arxiv.org/abs/2609.09745)）来自 [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)。条件扩散生成连续体机器人多模态配置空间路径；障碍物图网络+解析微分运动学；混合测试集 91% 成功率。

## 一句话定义

**0–4 障碍物混合测试集报告 91% 成功率。**

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
| **arXiv** | [2609.09745](https://arxiv.org/abs/2609.09745) |
| **项目页** | — |
| **代码** | https://github.com/qiuke-qiuke/pcc_diffuser |
| **开源** | **已开源** |
| **文内指标** | 0–4 障碍物混合测试集报告 91% 成功率。 |


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


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 测试集 | **0–4 障碍物** 混合场景 |
| 成功率 | **91%** |
| 机制 | 条件扩散生成 **配置空间多模态路径** + 障碍物 **图网络** + 解析 **微分运动学** |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md) 与来源归档；分障碍数的逐档成功率、规划耗时与基线以 **原文 PDF** 为准（[参考来源](#参考来源)）。
- **91% 是混合口径：** 0–4 障碍物合并统计，难度分布会拉高均值；比较时应索取 **按障碍数分档** 的那张表。
- **多解要单独看：** 本文卖点是一次给出 **多模态解**，只报单条路径成功率不足以体现——需看解的多样性与可行解覆盖。

## 与其他工作对比

| 对照路线 | 差异 |
|----------|------|
| 采样式规划器（RRT / PRM 类） | 单次给一条可行解，重采样才得第二条；本文由条件扩散 **一次生成多模态解集**。 |
| 优化式规划（轨迹优化） | 依赖初值、易陷单一局部解；本文用生成模型覆盖解空间的多个模态。 |
| 刚性臂运动规划 | 关节即自由度；连续体机器人须走 **PCC 等分段常曲率参数化**，本文配 **解析微分运动学** 保证配置–任务空间可导。 |
| 固定障碍编码 | 障碍数变化需重训或截断；本文用 **图网络** 编码障碍集合，天然适配 0–4 个的变长输入。 |
| [JEPA Policy](./paper-jepa-policy.md) | 同期反向取舍：那里为延迟去掉扩散，这里把扩散的 **多模态性** 当核心能力；选型时先问「要多解还是要低延迟」。 |

## 结论

**PccDiffuser 值得按「已开源」边界阅读：先核对仓库是否可跑，再引用文内成功率数字。**

1. 索引来源为公众号导读，实验细节以 arXiv PDF 为准。
2. 开源结论：**已开源**。
3. 选型时对照 [11 篇地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中相邻节点，避免重复造页。

## 关联页面

- [VLM 与操作 11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md)
- [模仿学习 (Imitation Learning)](../methods/imitation-learning.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [pccdiffuser_arxiv_2609_09745.md](../../sources/papers/pccdiffuser_arxiv_2609_09745.md)
- [wechat 11篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)
- [arXiv:2609.09745](https://arxiv.org/abs/2609.09745)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.09745)

- [GitHub](https://github.com/qiuke-qiuke/pcc_diffuser)
