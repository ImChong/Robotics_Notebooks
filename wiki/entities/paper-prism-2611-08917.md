---
type: entity
project_id: prism-2611-08917
tags:
  - paper
  - vla
status: draft
updated: 2026-11-20
arxiv: "2611.08917"
original:
  - ../../sources/papers/prism-2611-08917.md
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-trends-2026-11-h1.md
sources:
  - ../../sources/papers/prism-2611-08917.md
  - ../../sources/blogs/duomo_embodied_2026_11_first_half.md
summary: "新任务微调可能覆盖已有操作技能，不能只验证新任务结果。"
---
# PRISM：可保留技能的上下文原语路由与混合

**英文题名：** PRISM: Primitive Routing via In-context Skill Mixing for Retentive VLA Adaptation。官方原文：[arXiv:2611.08917v1](https://arxiv.org/abs/2611.08917v1)。当前为**部分原文核验节点**，需补齐实验与附录后才可标 complete。

## 一句话定义
长期技能保持与组合：新任务微调可能覆盖已有操作技能，不能只验证新任务结果。

## 英文缩写速查
| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉语言动作 |
| VLM | Vision-Language Model | 视觉语言模型 |
| OOD | Out of Distribution | 分布外评测 |

## 为什么重要
新任务微调可能覆盖已有操作技能，不能只验证新任务结果。 论文的价值在于将这一失败模式单独建模，而非依靠笼统的端到端指标解释。

## 核心机制
1. **步骤 1：** 原子任务训练低秩技能适配器。
2. **步骤 2：** 冻结原子技能库。
3. **步骤 3：** 视觉语言上下文送入路由器。
4. **步骤 4：** 组合技能与残差分支。
5. **步骤 5：** 预测当前动作块。

**研究边界：** 数据/动作接口的实际定义与可运行性，须从论文附录及实现进一步确认。

## 流程总览
以下仅概括已核验的论文方法步骤，不意味着所有代码模块均已公开。
```mermaid
flowchart LR
  A0["原子任务训练低秩技能适配器"]
  A0 --> A1["冻结原子技能库"]
  A1 --> A2["视觉语言上下文送入路由器"]
  A2 --> A3["组合技能与残差分支"]
  A3 --> A4["预测当前动作块"]
```

## 工程实践
- **任务对象：** 长期技能保持与组合。
- **论文版本：** v1；同一 arXiv ID 的后续版本不重复建页。
- **开源核对：** 当前未核验到可运行官方仓库；不能据此断言论文未来不会开源。
- **工程操作：** 先明确输入观测、动作接口、模型训练信号与验证任务；本文的结果不能直接转用到另一种本体。

## 源码运行时序图
**不适用**：截至本次检查未核实可运行官方源码入口；禁止由论文方法虚构函数和脚本调用。

## 实验与评测
论文研究 RoboCasa365、组合任务和 AgileX PIPER 真机，强调旧技能保持与新组合能力，需要对齐原子/复合协议。

**完整性说明：** 尚未逐表对照 v1 全文与补充材料，不制造精确数字，也不宣称独立复现。

## 局限与风险
技能库质量、路由失误以及不同原语间的动作接口不一致会限制迁移。

此外，本次未完成论文所有附录及统计协议逐项核对，项目主页/仓库状态可能更新；原文优先级高于二次传播解读。

## 与其他工作对比
- 在[九篇研究趋势地图](../overview/embodied-trends-2026-11-h1.md)中按“技术介入层”定位，不能比较跨论文不同任务、机器人和数据集的绝对指标。
- 相关基础：[VLA](../methods/vla.md)、[机器人操作](../tasks/manipulation.md)。

## 结论
**这篇工作的主要贡献是：长期技能保持与组合。**

- 关键机制：冻结原子技能库。
- 主要输出：预测当前动作块。
- 实证范围：论文研究 RoboCasa365、组合任务和 AgileX PIPER 真机，强调旧技能保持与新组合能力，需要对齐原子/复合协议。
- 局限与复现前提：技能库质量、路由失误以及不同原语间的动作接口不一致会限制迁移。
- 下一步：核对原文全部表格、附录以及代码依赖，在完成前维持 draft。

## 关联页面
- [VLA](../methods/vla.md)
- [机器人操作](../tasks/manipulation.md)
- [2026.11 上半月九篇论文技术图](../overview/embodied-trends-2026-11-h1.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)

## 参考来源
- [论文来源核对](../../sources/papers/prism-2611-08917.md)
- [策展资料](../../sources/blogs/duomo_embodied_2026_11_first_half.md)
- [arXiv 指定版本](https://arxiv.org/abs/2611.08917v1)

## 推荐继续阅读
- [指定版本 arXiv HTML](https://arxiv.org/html/2611.08917v1)
- [VLA 纵深路线](../../roadmap/depth-vla.md)
