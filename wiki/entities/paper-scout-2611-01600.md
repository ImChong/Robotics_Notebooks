---
type: entity
project_id: scout-2611-01600
tags:
  - paper
  - vla
status: draft
updated: 2026-11-20
arxiv: "2611.01600"
original:
  - ../../sources/papers/scout-2611-01600.md
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-trends-2026-11-h1.md
sources:
  - ../../sources/papers/scout-2611-01600.md
  - ../../sources/blogs/duomo_embodied_2026_11_first_half.md
summary: "货架操作常需先移除阻挡物或改变视角，不能仅选择当前可见的抓取动作。"
---
# SCOUT：遮挡场景的对象中心世界模型规划

**英文题名：** Scout: Planning under Occlusion via Object-Centric World Model Rollouts。官方原文：[arXiv:2611.01600v1](https://arxiv.org/abs/2611.01600v1)。当前为**部分原文核验节点**，需补齐实验与附录后才可标 complete。

## 一句话定义
遮挡物体探索与预测规划：货架操作常需先移除阻挡物或改变视角，不能仅选择当前可见的抓取动作。

## 英文缩写速查
| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉语言动作 |
| VLM | Vision-Language Model | 视觉语言模型 |
| OOD | Out of Distribution | 分布外评测 |

## 为什么重要
货架操作常需先移除阻挡物或改变视角，不能仅选择当前可见的抓取动作。 论文的价值在于将这一失败模式单独建模，而非依靠笼统的端到端指标解释。

## 核心机制
1. **步骤 1：** 货架场景观察与隐藏状态估计。
2. **步骤 2：** 对象中心关系和可达性表示。
3. **步骤 3：** 产生探索和操作候选动作。
4. **步骤 4：** 世界模型预测候选动作后果。
5. **步骤 5：** 选择动作并重新感知。

**研究边界：** 数据/动作接口的实际定义与可运行性，须从论文附录及实现进一步确认。

## 流程总览
以下仅概括已核验的论文方法步骤，不意味着所有代码模块均已公开。
```mermaid
flowchart LR
  A0["货架场景观察与隐藏状态估计"]
  A0 --> A1["对象中心关系和可达性表示"]
  A1 --> A2["产生探索和操作候选动作"]
  A2 --> A3["世界模型预测候选动作后果"]
  A3 --> A4["选择动作并重新感知"]
```

## 工程实践
- **任务对象：** 遮挡物体探索与预测规划。
- **论文版本：** v1；同一 arXiv ID 的后续版本不重复建页。
- **开源核对：** 当前未核验到可运行官方仓库；不能据此断言论文未来不会开源。
- **工程操作：** 先明确输入观测、动作接口、模型训练信号与验证任务；本文的结果不能直接转用到另一种本体。

## 源码运行时序图
**不适用**：截至本次检查未核实可运行官方源码入口；禁止由论文方法虚构函数和脚本调用。

## 实验与评测
论文展示货架遮挡下的 Reveal 和 Placement 任务与真实环境实验，尚未完整核对全部消融数字。

**完整性说明：** 尚未逐表对照 v1 全文与补充材料，不制造精确数字，也不宣称独立复现。

## 局限与风险
隐藏状态估计、模型滚动误差以及真实执行偏差均可能累积影响规划质量。

此外，本次未完成论文所有附录及统计协议逐项核对，项目主页/仓库状态可能更新；原文优先级高于二次传播解读。

## 与其他工作对比
- 在[九篇研究趋势地图](../overview/embodied-trends-2026-11-h1.md)中按“技术介入层”定位，不能比较跨论文不同任务、机器人和数据集的绝对指标。
- 相关基础：[VLA](../methods/vla.md)、[机器人操作](../tasks/manipulation.md)。

## 结论
**这篇工作的主要贡献是：遮挡物体探索与预测规划。**

- 关键机制：对象中心关系和可达性表示。
- 主要输出：选择动作并重新感知。
- 实证范围：论文展示货架遮挡下的 Reveal 和 Placement 任务与真实环境实验，尚未完整核对全部消融数字。
- 局限与复现前提：隐藏状态估计、模型滚动误差以及真实执行偏差均可能累积影响规划质量。
- 下一步：核对原文全部表格、附录以及代码依赖，在完成前维持 draft。

## 关联页面
- [VLA](../methods/vla.md)
- [机器人操作](../tasks/manipulation.md)
- [2026.11 上半月九篇论文技术图](../overview/embodied-trends-2026-11-h1.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)

## 参考来源
- [论文来源核对](../../sources/papers/scout-2611-01600.md)
- [策展资料](../../sources/blogs/duomo_embodied_2026_11_first_half.md)
- [arXiv 指定版本](https://arxiv.org/abs/2611.01600v1)

## 推荐继续阅读
- [指定版本 arXiv HTML](https://arxiv.org/html/2611.01600v1)
- [VLA 纵深路线](../../roadmap/depth-vla.md)
