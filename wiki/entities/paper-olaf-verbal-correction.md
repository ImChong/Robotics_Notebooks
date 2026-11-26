---
type: entity
tags: [paper, agentic-robots, robot-learning, microsoft, ut-austin]
status: complete
updated: 2026-11-16
project_id: olaf-verbal-correction
arxiv: "2310.17555"
project: "https://ut-austin-rpl.github.io/olaf/"
original:
  - "../../sources/papers/agentic_2310_17555.md"
sources:
  - "../../sources/papers/agentic_2310_17555.md"
  - "../../sources/sites/agentic-olaf-verbal-correction.md"
summary: "OLAF 将用户停止机器人时给出的语言纠错转为介入前短窗口的动作标签，并用行为克隆更新视觉运动策略。"
---

# OLAF：从语言纠错重标注动作并更新策略

## 一句话定义

OLAF 将用户停止机器人时给出的语言纠错转为介入前短窗口的动作标签，并用行为克隆更新视觉运动策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| OLAF | Operation-relabeled Learning with LAnguage Feedback | 语言反馈驱动动作重标注 |
| BC | Behavior Cloning | 在聚合数据上更新策略 |
| HIL | Human-in-the-loop | 人类观察、停止及可选接管 |

## 为什么重要

本工作说明了语言反馈如何进入持续适应闭环，并给出了可核对的学习对象、评测协议和实现边界。学习与工程选型时应先明确变化的是偏好规则、动作策略还是语言模型。

## 核心信息

| 项目 | 内容 |
|---|---|
| 论文 | Interactive Robot Learning from Verbal Correction |
| 固定版本 | [arXiv:2310.17555v1](https://arxiv.org/abs/2310.17555v1) |
| 机构 | 微软研究院、得克萨斯大学奥斯汀分校 |
| 原文核对 | v1 HTML §I–V、附录 §VI–IX 的工作流、提示词与策略超参数，Figure 1–6、Table I–VIII |
| 项目页 | [官方入口](https://ut-austin-rpl.github.io/olaf/) |

## 核心原理

用户观察初始策略，必要时停止并通过键盘输入纠错，可选遥操作接管。系统将停止前约 1–2 秒定义为 pre-intervention region；默认对 15 个时步应用从终止状态和纠错选出的修改量。

LLM 在单轴动作候选中充当 critic，选择 delta 后加到原动作；早期正常轨迹保留，再与原示范及可选接管数据聚合，用 BC 更新约 19M 参数的视觉 Transformer。推理时执行更新后的策略，LLM 查询发生在离线重标注阶段。

附录区分夹爪状态选择与 6D 动作选择，并用另一轮模型调用整理 JSON；格式错误最多重试 3 次。因此正文“一次查询覆盖一个窗口”的含义是无需逐时步重查，不应解读为整个软件工作流只有一次 API 请求。

## 流程总览

依据固定版本方法与系统说明整理：

```mermaid
flowchart LR
  policy["执行初始视觉运动策略"] --> stop["用户停止并给出语言纠错"]
  stop --> window["截取介入前短窗口与状态"]
  window --> critic["LLM 从单轴候选中选择修改量"]
  critic --> labels["修改量加回原动作并重标注窗口"]
  labels --> aggregate["与原示范及可选接管轨迹聚合"]
  aggregate --> bc["行为克隆更新策略权重"]
  bc --> policy
```

## 源码运行时序图

**不适用（当前可获取资源）**：项目页 Code 链接为空，尚无可核验的 OLAF 官方运行入口。公开上游 robomimic 不能替代本项目动作重标注实现，故仅绘制论文方法图。

## 工程实践

实现前应先对齐停止时刻、介入前窗口、状态坐标系与动作尺度；原文存在仿真交互预算的正文/附录差异。当前归档可用于方法分析，完整源码复现仍受官方资源开放范围限制。

## 实验与评测

以下均为论文作者报告，非本库复现实测。

| 实验 | BC 对照 | OLAF | 条件 |
|---|---|---|---|
| Square 仿真，只用语言 | 41.0 ± 2.2% | 59.0 ± 7.1% | 长语言反馈，三种子均值与标准差，Table I |
| 真机 PickPlace-Bin | 35.3% | 73.5% | 语言加物理介入，Table III |
| 真机 PickPlace-Drawer-Basket | 52.9% | 70.6% | 同上 |

仿真四任务每个初始示范 M=50，正文实验段写 N=100 交互轨迹，但附录 Table VII 写 50；该预算差异保留待核，不自行统一。仿真训练 1000 epochs，每 100 epochs 评估 50 次，比较种子内最佳 checkpoint。真机每任务 M=40、N=80，单种子、最后一轮 checkpoint 评估 34 次。不能把真机结果说成纯语言纠错或语音识别实验。

## 结论

OLAF 将用户停止机器人时给出的语言纠错转为介入前短窗口的动作标签，并用行为克隆更新视觉运动策略。应连同其数据、执行器和反馈预算一起评估。

1. 纠错窗口和人类反应延迟应一起校准。
2. 比较时区分仅语言与语言加物理介入。
3. 对齐状态文本、动作坐标系、delta 尺度和候选动作。

## 与其他工作对比

与 [PAFF](paper-paff-2212-07398.md) 重标注指令相比，OLAF 根据人类反馈修改动作标签；与 [LMPC](paper-lmpc-2402-11450.md) 的语言模型微调相比，OLAF 更新视觉运动策略。缩写 OLAF 与本库 Disney 角色机器人是不同研究。

## 局限与风险

- LLM 需要文本化状态和手工任务提示词；仿真使用真值状态，真机使用 DOPE 物体位姿估计。
- 同一个 delta 作用于整个介入前窗口是假设，较长或变化明显的窗口可能重标注错误。
- 实验语言通过键盘输入；不能据此声称语音识别与学习链已验证。
- 官方项目页的 Code 锚点 href 为空，核查时未提供可获取的本项目运行实现；不将上游 robomimic 当作 OLAF 官方完整代码。

## 关联页面

- [模仿学习](../methods/imitation-learning.md) — 延伸阅读，从示范学习策略。
- [Agentic Robots 综述](paper-towards-agentic-robots-survey.md)
- [VLA](../methods/vla.md)
- [VLA 纵深路线](../../roadmap/depth-vla.md)

## 参考来源

- [论文核读归档](../../sources/papers/agentic_2310_17555.md)
- [项目页核查](../../sources/sites/agentic-olaf-verbal-correction.md)

## 推荐继续阅读

- [固定版本原文](https://arxiv.org/html/2310.17555v1)
