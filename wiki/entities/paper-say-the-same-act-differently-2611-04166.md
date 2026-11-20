---
type: entity
project_id: say-the-same-act-differently-2611-04166
tags:
  - paper
  - vla
status: draft
updated: 2026-11-20
arxiv: "2611.04166"
original:
  - ../../sources/papers/say-the-same-act-differently-2611-04166.md
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-trends-2026-11-h1.md
sources:
  - ../../sources/papers/say-the-same-act-differently-2611-04166.md
  - ../../sources/blogs/duomo_embodied_2026_11_first_half.md
summary: "模型说出相同的解释，并不证明连续动作会对视觉扰动保持稳定。"
---
# Say the Same, Act Differently：解释不变、行为却变的 VLA 诊断

**英文题名：** Say the Same, Act Differently。官方原文：[arXiv:2611.04166v1](https://arxiv.org/abs/2611.04166v1)。当前为**部分原文核验节点**，需补齐实验与附录后才可标 complete。

## 一句话定义
语言与动作一致性安全诊断：模型说出相同的解释，并不证明连续动作会对视觉扰动保持稳定。

## 英文缩写速查
| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| VLA | Vision-Language-Action | 视觉语言动作 |
| VLM | Vision-Language Model | 视觉语言模型 |
| OOD | Out of Distribution | 分布外评测 |

## 为什么重要
模型说出相同的解释，并不证明连续动作会对视觉扰动保持稳定。 论文的价值在于将这一失败模式单独建模，而非依靠笼统的端到端指标解释。

## 核心机制
1. **步骤 1：** 采集 VLA 文本与动作输出。
2. **步骤 2：** 找出文本损失与动作敏感方向。
3. **步骤 3：** 在文本近中性子空间施加视觉扰动。
4. **步骤 4：** 观察动作偏移。
5. **步骤 5：** 比较说法稳定与控制风险。

**研究边界：** 这是对 VLA 解释与动作失配的诊断工作，而不是已验证的新安全控制器。

## 流程总览
以下仅概括已核验的论文方法步骤，不意味着所有代码模块均已公开。
```mermaid
flowchart LR
  A0["采集 VLA 文本与动作输出"]
  A0 --> A1["找出文本损失与动作敏感方向"]
  A1 --> A2["在文本近中性子空间施加视觉扰动"]
  A2 --> A3["观察动作偏移"]
  A3 --> A4["比较说法稳定与控制风险"]
```

## 工程实践
- **任务对象：** 语言与动作一致性安全诊断。
- **论文版本：** v1；同一 arXiv ID 的后续版本不重复建页。
- **开源核对：** 当前未核验到可运行官方仓库；不能据此断言论文未来不会开源。
- **工程操作：** 先明确输入观测、动作接口、模型训练信号与验证任务；本文的结果不能直接转用到另一种本体。

## 源码运行时序图
**不适用**：截至本次检查未核实可运行官方源码入口；禁止由论文方法虚构函数和脚本调用。

## 实验与评测
论文在 Alpamayo 驾驶和 InstructVLA 操作设定中讨论文字基本不变时的行为变化；所示碰撞属于研究定义的离线攻击协议。

**完整性说明：** 尚未逐表对照 v1 全文与补充材料，不制造精确数字，也不宣称独立复现。

## 局限与风险
对抗扰动成功率不能当成自然交通场景的事故率或真实机器人故障频率。

此外，本次未完成论文所有附录及统计协议逐项核对，项目主页/仓库状态可能更新；原文优先级高于二次传播解读。

## 与其他工作对比
- 在[九篇研究趋势地图](../overview/embodied-trends-2026-11-h1.md)中按“技术介入层”定位，不能比较跨论文不同任务、机器人和数据集的绝对指标。
- 相关基础：[VLA](../methods/vla.md)、[机器人操作](../tasks/manipulation.md)。

## 结论
**这篇工作的主要贡献是：语言与动作一致性安全诊断。**

- 关键机制：找出文本损失与动作敏感方向。
- 主要输出：比较说法稳定与控制风险。
- 实证范围：论文在 Alpamayo 驾驶和 InstructVLA 操作设定中讨论文字基本不变时的行为变化；所示碰撞属于研究定义的离线攻击协议。
- 局限与复现前提：对抗扰动成功率不能当成自然交通场景的事故率或真实机器人故障频率。
- 下一步：核对原文全部表格、附录以及代码依赖，在完成前维持 draft。

## 关联页面
- [VLA](../methods/vla.md)
- [机器人操作](../tasks/manipulation.md)
- [2026.11 上半月九篇论文技术图](../overview/embodied-trends-2026-11-h1.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)

## 参考来源
- [论文来源核对](../../sources/papers/say-the-same-act-differently-2611-04166.md)
- [策展资料](../../sources/blogs/duomo_embodied_2026_11_first_half.md)
- [arXiv 指定版本](https://arxiv.org/abs/2611.04166v1)

## 推荐继续阅读
- [指定版本 arXiv HTML](https://arxiv.org/html/2611.04166v1)
- [VLA 纵深路线](../../roadmap/depth-vla.md)
