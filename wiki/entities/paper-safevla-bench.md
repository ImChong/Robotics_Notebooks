---
type: entity
tags: [paper, manipulation, vla]
status: complete
updated: 2026-10-06
arxiv: "2606.00773"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../overview/frontier-manipulation-2026-09-28-10-02.md
sources:
  - ../../sources/blogs/frontier_manipulation_2026_09_28_10_02.md
summary: "SafeVLA-Bench 把 Manipulation Safety 写成 Signal Temporal Logic （ STL ）不变量，在原有 LIBERO 、 RoboCasa-365 rollout 上进行 post-hoc safety evaluation，不需要重新设计整个 Benchmark。除了 Success，还报告 Safety R"
---

# SafeVLA-Bench：视觉-语言-动作模型成功率与安全性差距评测基准

**SafeVLA-Bench: A Benchmark for the Success-Safety Gap in Vision- Language-Action Models**（[arXiv:2606.00773](https://arxiv.org/abs/2606.00773)）聚焦一个机器人学习/控制缺口：当前 VLA Benchmark 主要看任务有没有完成，但“成功”不代表执行过程安全。例如机器 人最后把杯子放到了正确位置，但过程中可能碰倒旁边物体、施加过大的接触力、让抓取物体失稳，甚至 发生机器人 self-contact，这些通常不会反映在最终 success rate 中。文章列出的机构：University of Notre Dame、University of Pennsylvania。

## 一句话定义

SafeVLA-Bench：视觉-语言-动作模型成功率与安全性差距评测基准是一项针对「当前 VLA Benchmark 主要看任务有没有完成，但“成功”不代表执行过程安全。例如机器 人最后把杯子放到了正确位置，但过程中可能碰倒旁边物体、施加过大的接触力、让」提出的研究；其贡献应理解为一条方法或评测线索，具体实现细节以论文原文为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉、语言与动作联合建模的策略接口 |
| RL | Reinforcement Learning | 通过环境回报优化控制策略 |
| OOD | Out-of-Distribution | 训练分布之外的测试条件 |
| STL | Signal Temporal Logic | 描述时序安全约束的逻辑形式 |

## 为什么重要

- **问题定位：** 当前 VLA Benchmark 主要看任务有没有完成，但“成功”不代表执行过程安全。例如机器 人最后把杯子放到了正确位置，但过程中可能碰倒旁边物体、施加过大的接触力、让抓取物体失稳，甚至 发生机器人 self-contact，这些通常不会反映在最终 success rate 中
- **方法侧重点：** SafeVLA-Bench 把 Manipulation Safety 写成 Signal Temporal Logic （ STL ）不变量，在原有 LIBERO 、 RoboCasa-365 rollout 上进行 post-hoc safety evaluation，不需要重新设计整个 Benchmark。除了 Success，还报告 Safety Rate 、 Success-but-Unsafe Rate，以及描述最严重违规程度的 Violation Severity Index。实验发现，即使平均成功率超过 90% 的 15 个 tabletop policy，仍有 18–28% unsafe episode； RoboCasa-365 中 38–56% 的成功 rollout 至少违反一个安全条件。论文还展示了利用这些指标进行 Policy Post-Training 的案例。 【9.21-9.25前沿论文动态】Manipulation15 篇
- **阅读边界：** 这条工作适合与相关任务/算法并读；文章摘要不足以证明方法能直接迁移到另一机器人或应用场景。

## 方法栈与流程总览

```mermaid
flowchart LR
  P["问题：当前 VLA Benchmark 主要看任务有没有完成，但“成功”不代表执行过程安全。例如机器 人最后把杯子放到了正确位置，但过程中可能碰倒旁边物体、施加过大的接触力、让抓取物"] --> M["机制：SafeVLA-Bench 把 Manipulation Safety 写成 Signal Temporal Logic （ STL ）不变量，在原有 LIBERO 、 Rob"] --> E["结果：原文评测需核验"]
```

此图只归纳文章摘要中明确给出的「问题—方法」关系，不替代论文方法图或实际运行时序。

## 方法

文章归纳的关键机制是：SafeVLA-Bench 把 Manipulation Safety 写成 Signal Temporal Logic （ STL ）不变量，在原有 LIBERO 、 RoboCasa-365 rollout 上进行 post-hoc safety evaluation，不需要重新设计整个 Benchmark。除了 Success，还报告 Safety Rate 、 Success-but-Unsafe Rate，以及描述最严重违规程度的 Violation Severity Index。实验发现，即使平均成功率超过 90% 的 15 个 tabletop policy，仍有 18–28% unsafe episode； RoboCasa-365 中 38–56% 的成功 rollout 至少违反一个安全条件。论文还展示了利用这些指标进行 Policy Post-Training 的案例。 【9.21-9.25前沿论文动态】Manipulation15 篇

**输入与输出。** 文章所述输入包括任务相关的视觉、本体状态、动作示范或控制指令，取决于原文具体设定；输出是论文提出的策略、表示、评测协议或控制机制。由于附件未给出完整实验协议，实际 observation/action 定义与控制频率应从论文方法节核查。

## 实验与评测

- **附件提供的证据：** SafeVLA-Bench 把 Manipulation Safety 写成 Signal Temporal Logic （ STL ）不变量，在原有 LIBERO 、 RoboCasa-365 rollout 上进行 post-hoc safety evaluation，不需要重新设计整个 Benchmark。除了 Success，还报告 Safety Rate 、 Success-but-Unsafe Rate，以及描述最严重违规程度的 Violation Severity Index。实验发现，即使平均成功率超过 90% 的 15 个 tabletop policy，仍有 18–28% unsafe episode； RoboCasa-365 中 38–56% 的成功 rollout 至少违反一个安全条件。论文还展示了利用这些指标进行 Policy Post-Training 的案例。 【9.21-9.25前沿论文动态】Manipulation15 篇
- **复核要点：** 先确认论文中的机器人/仿真器、训练数据与 baseline，再将成功率或误差等数字按相同任务协议比较。
- **定量结果：** 若附件段落未明确报告数值，不在此补造；点击 arXiv 原文查看完整表格与消融。

## 源码运行时序图

**源码运行时序图不适用。** 本次综述 PDF 未附该论文的官方项目页或代码仓库链接，当前无法核验可运行训练/推理入口；这不等于判定其未开源。

## 工程实践

| 环节 | 复现时需要核对 |
|------|----------------|
| 观测与动作 | 传感器、状态维度、动作表示及控制频率是否与目标机器人一致 |
| 训练/推理 | 是否需要仿真、预训练模型、额外传感器或在线优化器 |
| 评测 | 场景划分、baseline、失败定义与真实机器人验证条件 |
| 源码/项目资源 | 综述未附官方项目页或运行仓库，开源状态未核验。 |

## 结论

**一句话总判：** SafeVLA-Bench 把 Manipulation Safety 写成 Signal Temporal Logic （ STL ）不变量，在原有 LIBERO 、 RoboCasa-365 rollout 上进行 post-hoc safety evaluation，不需要重新设计整个 Benchmark。除了 Success，还报告 Safety Rate 、 Success-but-Unsafe Rate，以及描述最严重违规程度的 Violation Severity Index。实验发现，即使平均成功率超过 90% 的 15 个 tabletop policy，仍有 18–28% unsafe episode； RoboCasa-365 中 38–56% 的成功 rollout 至少违反一个安全条件。论文还展示了利用这些指标进行 Policy Post-Training 的案例。 【9.21-9.25前沿论文动态】Manipulation15 篇从文章摘要能确认研究动机与方法主线，但性能与可复现边界需要以原文核实。

1. **先核对问题设定。** 确认其观测、机器人本体与动作接口是否匹配目标应用。
2. **把方法拆成可验证模块。** 逐一查明表示、策略、控制器或数据处理的作用，避免只按论文命名判断。
3. **复现优先看消融与失败条件。** 检查增益来自关键模块还是数据规模、额外传感器或更宽松的评测设置。
4. **不要把文章摘要当成开源状态证明。** 当前详情只记录文章所载信息；代码与数据状态未在本次资料中核验。

## 局限与风险

- 当前总结来自两篇综述 PDF，未对每篇原文逐项复核；实验数字、模型版本和技术细节应以 arXiv 页面/正文为准。
- 综述没有提供本条论文的项目页或代码仓库链接，因此不推断「已开源」或「未开源」，也不绘制源码运行时序图。
- 若方法依赖专用传感器、动作先验、预训练 checkpoint 或定制低层控制器，迁移成本需单独评估。

## 与其他工作对比

| 维度 | 本工作 | 阅读时对照 |
|------|---------|-------------|
| 研究缺口 | 当前 VLA Benchmark 主要看任务有没有完成，但“成功”不代表执行过程安全。例如机器 人最后把杯子放到了正确位置，但过程中可能碰倒旁边物体、施加过大的接触力、让抓取物体失稳，甚至 发生机器人 | 相关任务页中常用方法的输入与输出 |
| 方法主线 | SafeVLA-Bench 把 Manipulation Safety 写成 Signal Temporal Logic （ STL ）不变量，在原有 LIBERO 、 RoboCasa-365 ro | 直接策略、模型/规划或学习式控制基线 |
| 证据边界 | 综述提要；以原文评测表和消融为准 | 相同机器人、相同场景、相同成功标准 |

## 关联页面

- [【9.28–10.2 前沿论文动态】Manipulation](../overview/frontier-manipulation-2026-09-28-10-02.md)
- [Manipulation](../tasks/manipulation.md)
- [Vision-Language-Action](../methods/vla.md)

## 参考来源

- [【9.28–10.2 前沿论文动态】Manipulation 综述条目](../../sources/blogs/frontier_manipulation_2026_09_28_10_02.md)
- [arXiv:2606.00773](https://arxiv.org/abs/2606.00773)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2606.00773)
- [Manipulation任务页](../tasks/manipulation.md)
