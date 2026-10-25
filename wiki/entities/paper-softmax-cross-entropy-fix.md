---
type: entity
tags: [paper, numerical-stability, softmax, cross-entropy, knowledge-distillation, machine-learning]
status: complete
updated: 2026-10-25
arxiv: "2610.31642"
project_id: a-fix-softmax-cross-entropy
venue: "arXiv 2026"
related:
  - ../methods/awr.md
  - ./paper-scaling-laws-neural-language-models.md
sources:
  - ../../sources/papers/softmax_cross_entropy_fix_arxiv_2610_31642.md
summary: "Barron、Gnanapragasam 与 Tschannen（Google DeepMind，arXiv:2610.31642）指出标准 softmax / cross-entropy 的高置信度梯度坍缩和蒸馏近似相消问题，并给出 JAX 修正版。"
---

# A Fix for Softmax and Cross-Entropy

**A Fix for Softmax and Cross-Entropy**（Jonathan T. Barron、Dan Gnanapragasam、Michael Tschannen，Google DeepMind，[arXiv:2610.31642](https://arxiv.org/abs/2610.31642)，2026）分析常见 softmax、log-softmax 与交叉熵实现中的浮点精度问题，并提供直接计算 loss 和梯度的修正版。

## 一句话定义

**在高置信度分类与学生教师 logits 接近时，避免浮点舍入把正确梯度抹掉或制造伪梯度。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CE | Cross-Entropy | 分类与蒸馏常用损失；本文修正其数值实现 |
| KL | Kullback–Leibler Divergence | 学生与教师分布差异；蒸馏实验中理论值应为零 |
| bf16 / fp32 / fp64 | bfloat16 / 32-bit / 64-bit Floating Point | 论文对比的三种浮点精度 |
| LSE | Log-Sum-Exp | 对非目标类 logits 求和以得到目标 margin |
| JAX | JAX | 论文给出修正版参考实现所用的数值计算框架 |

## 两类问题

| 问题 | 发生位置 | 标准实现的误差 |
|------|----------|----------------|
| **Softmax Collapse** | 正确类别 logit 远大于其他类别 | 计算 \(1+e^{-m}\) 时，小项被舍入掉；目标概率变为 1、目标梯度 \(q_t-1\) 变为 0，但其他类别梯度仍非零，因此梯度和不为 0 |
| **Softmax Cancellation** | 软目标交叉熵 / 知识蒸馏，学生与教师 logits 接近 | 分别计算概率后做 \(q-p\) 会损失有效位；即使 logits 相等，也可能留下非零梯度 |

论文报告 Softmax Collapse 的 logit-margin 精度悬崖约为 bf16 的 5.5、fp32 的 16.6、fp64 的 36.7。这个现象不是普通的指数溢出：减最大 logit 的常见技巧避免了溢出，但并未消除小梯度在加减 1 时被舍入的问题。

## 方法与修正思路

- **硬标签交叉熵：** 对目标类之外的 logits 计算 log-sum-exp，得到目标 margin \(m\)。损失写作 \(\operatorname{softplus}(-m)\)，目标梯度直接写成 \(-e^{-m}/(1+e^{-m})\)，不再从 \(q_t-1\) 相减得到。
- **log-softmax / softmax：** 排除最大 logit后计算 margin，并用 softplus 组合结果；最大值的指数项不再与很小的项直接相加。
- **软目标交叉熵：** 在 log 空间用学生和教师 logits 差计算概率比，以 `expm1`、`log1p` 保留接近相等时的微小变化，再对梯度和作浮点漂移校正。

原文给出 JAX 函数作为实现示例；本论文页面未列出独立代码仓库。

## 评测结果

- 在 4 个 grokking 基准（模加法 1 层 Transformer、平方/立方模运算 2 层 MLP）中，作者报告 fp32 修正版匹配 fp64 标准交叉熵，并达到 100% 测试准确率。
- 在 \(a^2+b^2\bmod 97\) 且无 weight decay 的实验中，标准 fp32 实现的测试准确率在 Slingshot 尖峰后从 61% 降到 23%；修正版继续到 85%。
- 在 \(a+b\bmod 31\) 的精度对照中，修正版于 fp32、bf16、fp64 和所测 Adam \(\epsilon\) 设置下达到 98.9–100% 测试准确率。
- 在软目标蒸馏测试中，学生从教师权重初始化时理论梯度应为零。作者报告标准实现的残余梯度会使 bf16 学生偏离教师；修正版在各精度下保持 KL 为零、测试准确率为 100%。

## 对比

与标准公式相比，修正版显式保留接近 0 的梯度项，并在蒸馏中直接利用 logits 差，避免相近概率相减。论文也比较零和梯度投影与 StableMax；在所测 grokking 设置中，作者报告修正版的准确率更高、margin 更接近 Adam 的理论平衡值。这些比较限于论文实验设定。

## 结论

修正版消除论文指出的两类数值误差，并改善其 grokking 与蒸馏实验；适用于机器人训练与否仍需在具体任务中验证。

这些结果来自小规模合成任务。它们说明数值公式会影响优化行为，但不直接证明机器人策略训练会遇到同样的故障或获得相同收益。

## 对机器人学习的启发

交叉熵可能用于离散动作预测、动作 token 预测、分类辅助损失和策略蒸馏。检查异常时可同时看 loss、logit margin、目标类梯度、整行梯度和及精度类型。高置信度后 loss 归零或蒸馏学生持续偏离教师，可以作为排查线索；是否由此问题引起需在目标框架和任务上复现。

## 关联页面

- [AWR](../methods/awr.md) — 加权交叉熵用于策略更新的机器人学习方法
- [Scaling Laws for Neural Language Models](./paper-scaling-laws-neural-language-models.md) — 交叉熵作为语言模型预训练评测损失的背景

## 参考来源

- [arXiv HTML 全文](https://arxiv.org/html/2610.31642)
- [arXiv PDF](https://arxiv.org/pdf/2610.31642)
- [论文来源归档](../../sources/papers/softmax_cross_entropy_fix_arxiv_2610_31642.md)
