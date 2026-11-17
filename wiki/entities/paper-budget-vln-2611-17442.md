---
type: entity
project_id: paper-budget-vln-2611-17442
tags: [paper,vision-language-navigation,exploration,planning]
status: complete
updated: 2026-11-17
arxiv: "2611.17442"
original:
  - ../../sources/papers/budget-vln-2611-17442.md
sources:
  - ../../sources/papers/budget-vln-2611-17442.md
related:
  - ./paper-wato-2611-17626.md
  - ./paper-pear-robot-use-rl.md
  - ./paper-agentic-teacher-2611-17394.md
summary: "用剩余步数预算指导视觉语言导航中的多目标搜索，采用部分可观测房间级 Bellman 教师监督，而不只模仿全知 TSP 路线。"
---

# Racing the Clock: Teaching VLNs Budget-Conditioned Object Search｜Racing the Clock：按预算进行视觉语言目标搜索

## 一句话定义

**Racing the Clock：按预算进行视觉语言目标搜索**：用剩余步数预算指导视觉语言导航中的多目标搜索，采用部分可观测房间级 Bellman 教师监督，而不只模仿全知 TSP 路线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLN | Vision-Language Navigation | 视觉语言导航 |
| TSP | Traveling Salesperson Problem | 已知目标位置下的全局巡回路径基线 |
| HM3D | Habitat-Matterport 3D Dataset | 室内扫描场景导航数据集 |

## 为什么重要

现实导航机器人搜索目标时不知道所有物品的精确位置，还受剩余时间与动作预算限制；全知专家即便轨迹最优，也可能不是最适合学生模仿的老师。

## 核心信息

| 项目 | 核查来源 |
|---|---|
| 作者或机构 | 原文作者机构，见 arXiv v1 |
| 原文 | [一手全文/原始新闻稿](https://arxiv.org/html/2611.17442v1) |
| 代码状态 | 未核实与本研究/设备对应的可运行代码仓库 |
| 核查版本 | arXiv:2611.17442v1 |

## 核心原理

输入目标类别、剩余动作预算与已探索房间结构；房间级 Bellman 规划在部分可观测条件下利用已见门/房间与空间语义先验分配搜索动作。将专家轨迹蒸馏给 VLN 学生，并以可达目标发现覆盖率在不同预算下测试。与知道所有目标坐标的对象级全知教师和 TSP 专家比较。

## 流程总览

```mermaid
flowchart LR
  A[目标类别与剩余步数]
  B[已探索房间与语义先验]
  C[部分可观测房间级 Bellman 教师]
  D[预算条件专家轨迹]
  E[VLN 学生策略蒸馏]
  F[预算自适应目标搜索]
  A --> C
  B --> C
  C --> D
  D --> E
  E --> F
```

上述流程图是基于原始资料整理的方法主干，不代表已复现作者模型。

## 源码运行时序图

**不适用：** 已核对 arXiv v1，但未确认本研究官方可运行训练代码及 CLI 入口，不以规划公式冒充软件流程。

## 工程实践

1. 训练/测试应固定每回合目标集合、剩余动作 budget 与发现可达目标的定义。
2. 区分教师轨迹本身的总覆盖率和蒸馏后学生在部分可观测条件下的覆盖率。
3. 按 500、1000、2000 等预算分段报告成功率，校验场景划分和隐藏地图信息是否泄漏。

## 实验与评测

arXiv v1 摘要称 HM3D 留出场景 2000 步内搜索到 71.4% 的可达目标；同一 HTML v1 正文表 II 却列房间级学生为 72.7% ± 6.8。两处数字不一致，尚不能确定是平均口径还是修订遗漏，故并列保留。

## 与其他工作对比

对象级全知 TSP 教师路线并不保证学生表现更好；[WATO](./paper-wato-2611-17626.md) 同样通过规划结构改善学习，但约束在动力学轨迹而非空间信息可观测性。

## 结论

**用剩余步数预算指导视觉语言导航中的多目标搜索，采用部分可观测房间级 Bellman 教师监督，而不只模仿全知 TSP 路线。**

1. 教师的可见信息条件应尽量接近学生部署时的条件。
2. 对搜索策略显式传入剩余时间/动作预算。
3. 将 71.4% 和 72.7% 的原文矛盾保留为待核查事项。

## 局限与风险

房间抽象和先验场景语义可能与真实环境不符，较长预算下仍有地图记忆瓶颈；论文摘要与正文结果不一致需进一步作者确认。

## 关联页面

- [paper wato 2611 17626](./paper-wato-2611-17626.md)
- [paper pear 2611 18562](./paper-pear-robot-use-rl.md)
- [paper agentic teacher 2611 17394](./paper-agentic-teacher-2611-17394.md)
- [视觉语言导航](../tasks/vision-language-navigation.md)
- [自主探索](../tasks/autonomous-exploration.md)
- [navigation 路线](../../roadmap/depth-navigation.md)

## 参考来源

- [一手原文核对记录](../../sources/papers/budget-vln-2611-17442.md)

## 推荐继续阅读

- [一手正文](https://arxiv.org/html/2611.17442v1)
- [作者/扩展资料](https://arxiv.org/abs/2611.17442)

