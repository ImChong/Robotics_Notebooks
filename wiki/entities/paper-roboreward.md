---
type: entity
project_id: roboreward
tags: [paper, reward-modeling, progress-reward, vlm, robotics, reinforcement-learning, stanford, berkeley]
status: complete
updated: 2026-10-07
arxiv: "2601.00675"
project: https://crfm.stanford.edu/helm/robo-reward-bench/
related:
  - ../concepts/progress-reward-modeling.md
  - ../concepts/reward-design.md
  - ../methods/reinforcement-learning.md
  - ./paper-topreward.md
sources:
  - ../../sources/papers/roboreward_arxiv_2601_00675.md
  - ../../sources/sites/roboreward-benchmark.md
summary: "RoboReward（arXiv:2601.00675，Stanford/UC Berkeley）：用反事实重标注和时间裁剪从成功轨迹构造 negatives 与 near-misses，发布机器人奖励数据/基准及 4B/8B VLM；8B 真机 RL 优于 Gemini Robotics-ER 1.5 基线。"
---

# RoboReward：面向机器人的通用视觉语言奖励模型

**RoboReward** 是面向机器人策略学习的视觉语言奖励数据集、评测基准和模型系列，目标是从任务指令与机器人轨迹中判断完成程度，为奖励设计和强化学习提供自动信号。

## 一句话定义

**RoboReward 把真实机器人成功轨迹改造成包含失败与部分进度的训练样本，再训练视觉语言模型给机器人轨迹打奖励分。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 读取任务文本和视觉轨迹并输出评价 |
| RL | Reinforcement Learning | 使用奖励信号更新机器人策略 |
| OXE | Open X-Embodiment | 多机器人演示数据来源之一 |
| HF | Hugging Face | 发布数据和模型权重的平台 |

## 为什么重要

机器人奖励通常来自人工打标签或人工写规则：前者昂贵，后者脆弱。现有真实数据又常以成功轨迹为主，模型容易把“看起来像成功”误当作充分奖励。RoboReward 把失败、近失误和任务中途状态纳入数据，直接评测奖励模型能否判断进展，并在真机 RL 中验证奖励的下游用途。

## 核心信息

| 项 | 内容 |
|----|------|
| 作者 / 机构 | Tony Lee、Andrew Wagenmaker、Karl Pertsch、Percy Liang、Sergey Levine、Chelsea Finn；Stanford University、UC Berkeley |
| 论文 | [arXiv:2601.00675](https://arxiv.org/abs/2601.00675) |
| 项目 / 基准 | [HELM RoboReward Bench](https://crfm.stanford.edu/helm/robo-reward-bench/) |
| 数据 | [teetone/RoboReward](https://huggingface.co/datasets/teetone/RoboReward) |
| 权重 | [RoboReward-8B](https://huggingface.co/teetone/RoboReward-8B)；论文报告 4B/8B 两种规模 |
| 开源状态 | 数据和模型已公开；本次未发现官方训练代码链接 |

## 核心原理

1. **建立数据与基准：** 汇总 Open X-Embodiment 与 RoboArena 等真实机器人轨迹，统一评测不同视觉语言模型的奖励判断。
2. **补齐失败样本：** 对成功 episode 做反事实重标注构造 calibrated negatives / near-misses；时间裁剪则生成部分完成状态，避免成功样本独占标签分布。
3. **训练奖励模型：** 论文训练 4B 和 8B 参数模型，给任务轨迹分配完成度判断。公开 HF 8B 模型卡以视频和指令为输入，输出 1–5 的离散终局进度分。
4. **回到策略优化：** 将 8B 奖励模型用于真机 RL，以策略改进衡量 reward model 是否不仅会“打分”，还对学习有帮助。

### 流程总览

```flowchart LR
  corpora["OXE + RoboArena 真实轨迹"] --> aug["反事实重标注 + 时间裁剪"]
  aug --> set["成功 / 失败 / 部分进度数据"]
  set --> bench["奖励模型基准评测"]
  set --> train["微调 RoboReward 4B / 8B"]
  train --> score["任务视频 + 指令 → 奖励分"]
  score --> policy["真机 RL 策略更新"]
```

## 源码运行时序图

**不适用** — 项目资源页公开数据集、模型和 evaluation suite，但本次未找到官方训练/推理代码仓库链接；权重可用不代表训练源码已公开。HF 模型卡提供 Transformers 推理提示，读者可复用基础模型推理接口。

## 实验与评测

- **模型评测：** 多种开源与闭源 VLM 没有任何一个在所有任务都领先，说明通用语义理解不足以保证机器人奖励质量。
- **奖励判断：** RoboReward 4B/8B 在短时程机器人任务奖励判断上超过更大通用 VLM。
- **下游真机 RL：** 8B reward VLM 相比 Gemini Robotics-ER 1.5 改善策略学习，并缩小与人类奖励训练的性能差距。
- **数据补强的意义：** negatives、near-misses 和 partial progress 是主要数据设计；单纯扩大成功轨迹规模无法替代这些反例。

## 工程实践

| 使用场景 | 做法 |
|----------|------|
| 轨迹打分 | 固定任务指令和终局判据，再给 rollout 视频评分；模型卡定义 1–5 级输出 |
| 奖励接入 RL | 先检查奖励与真实任务成功率的相关性，再评估策略是否通过钻漏洞获取高分 |
| 数据治理 | 保留 near-miss 与失败类型，不要只用成功视频微调奖励 VLM |
| 复现 | 已有 HF 数据与 8B 权重；未发现官方训练代码，完整训练复现仍受限 |

## 局限与风险

- 终局离散分数不能自动提供每一步的精确信用分配；若直接用于长时程 RL，仍需处理奖励稀疏与时序定位。
- 数据来源和任务分布决定模型的覆盖边界；机器人本体或相机变化会影响视觉判分。
- 奖励模型可能被策略利用；下游 RL 增益不能替代独立的奖励校准与安全检查。
- 4B/8B 结论来自论文设定，不保证其他模型规模、数据集或任务上同样成立。

## 结论

**RoboReward 的关键贡献是把反例构造、奖励基准、开源权重和真机 RL 效用连成一条评估链，避免只凭 VLM 语言能力推断它能当机器人奖励。**

1. **先看数据分布：** 成功占多数会让奖励模型缺少失败和部分进展边界。
2. **反事实重标注 + 时间裁剪** 是制造负样本与中间进度的关键操作。
3. **模型分数不是终点：** 需用下游策略改进验证奖励信号的价值。
4. **公开资源有实用性：** 数据集和 8B 模型可访问，训练管线开放程度较低。
5. **部署防奖励投机：** 同时跟踪独立成功率、失败类型和人工抽检。

## 关联页面

- [过程奖励建模](../concepts/progress-reward-modeling.md) — 进度信号的输出形态与效用评测
- [Reward Design](../concepts/reward-design.md) — 与手工奖励和偏好奖励的关系
- [TOPReward](./paper-topreward.md) — 基于 VLM token 概率的奖励路线
- [Reinforcement Learning](../methods/reinforcement-learning.md) — 奖励接入策略优化

## 参考来源

- [RoboReward 论文归档](../../sources/papers/roboreward_arxiv_2601_00675.md)
- [RoboReward 资源页归档](../../sources/sites/roboreward-benchmark.md)

## 推荐继续阅读

- [HELM RoboReward Bench](https://crfm.stanford.edu/helm/robo-reward-bench/) — 基准与资源入口
- [RoboReward-8B 模型卡](https://huggingface.co/teetone/RoboReward-8B) — 评分接口和使用方式
