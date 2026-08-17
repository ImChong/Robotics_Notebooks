---
type: concept
tags:
  - progress-reward
  - reward-modeling
  - reinforcement-learning
  - vlm
  - evaluation
  - survey
  - northwestern
  - cmu
status: complete
updated: 2026-08-17
related:
  - ../entities/paper-progress-reward-modeling-survey.md
  - ../entities/paper-topreward.md
  - ../entities/paper-prm-as-a-judge.md
  - ../methods/reinforcement-learning.md
  - ../methods/imitation-learning.md
  - ../methods/vla.md
  - ../concepts/contact-rich-manipulation.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../queries/embodied-fm-taxonomy-loop.md
sources:
  - ../../sources/papers/progress_reward_modeling_survey_arxiv_2607_21655.md
  - ../../sources/repos/awesome-progress-models.md
  - ../../sources/papers/topreward_arxiv_2602_19313.md
summary: "过程奖励/进度模型：在终局成功之外估计任务是否在推进、停滞或回退；用接口三维×四种构造范式×保真/鲁棒/效用评测透镜阅读该领域。"
---

# 过程奖励建模（Progress Reward Modeling）

**过程奖励 / 进度模型** 回答执行中的问题：在当前目标下，机器人是在 **推进、停滞，还是回退**？它把「终局成功」之外的稠密、可比较信号用于 RL、监控、重排、过滤与恢复。

## 一句话定义

**把任务推进程度建成可查询的模型接口，使信用分配与行为评价不必等到 episode 结束。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PRM | Progress Reward Model | 过程/进度奖励模型统称 |
| VLM | Vision-Language Model | 冻结或指令微调进度打分的常见骨干 |
| Δ-progress | Progress delta | 转移级相对进度，而非绝对标量 |
| RLHF/RLAIF | RL from Human/AI Feedback | 偏好数据构造奖励的近亲路线 |
| ORM | Outcome Reward Model | 终局/结果奖励对照 |
| Awesome | Awesome-Progress-Models | 综述配套开源论文索引 |

## 为什么重要

- **长时程稀疏奖励** 下，终局 0/1 无法区分「慢但对」与「空转/回退」。
- **同一画面可对应不同进度**（重复拣放、已完成子目标被撤销）——进度是 **历史依赖的潜在任务状态**。
- 文献碎片化：同名方法可能输出成功概率、百分比、偏好序或奖励程序；需要统一读法。
- 工程上可复用：[Awesome-Progress-Models](https://github.com/sterzhang/Awesome-Progress-Models) 按综述结构维护可点进的论文画廊。

## 核心原理

### 接口三维（先当黑盒）

| 轴 | 选项 | 核心权衡 |
|----|------|----------|
| 当前任务状态 | 单帧 · 时序窗 · 关系比较 · 状态/API | 上下文越长越消歧，在线越贵 |
| 任务目标 | 语言 · 目标图/演示 · 程序/谓词 | 物理细节 vs 标注/特权状态成本 |
| 输出形态 | 标量 · Δ · 排序 · 可执行奖励代码 | 决定适监控、比较、规划还是 RL |

### 四种构造范式

| 范式 | 信号从哪来 | 长处 | 主要陷阱 |
|------|------------|------|----------|
| 冻结基础模型打分 | 图文相似、token 概率、上下文价值 | 零样本 | 语义先验 ≠ 已标定奖励；实例见 [TOPReward](../entities/paper-topreward.md) |
| 时序/相对监督 | 演示时间序、邻近、偏好 | 弱监督可规模化 | 「更晚」≠「更好」 |
| 指令微调进度预测 | 显式进度/成功/Δ/推理目标 | 专用能力 | 需覆盖失败与回退 |
| 程序化奖励 | LLM/VLM 生成代码与特征 | 可解释可改 | 只及于可用状态变量 |

### 评测透镜（分清主张）

1. **进度保真** — 标定、时间一致性、相对序、目标接地、可否拒答。
2. **鲁棒/泛化** — 未见任务、视角、跨 embodiment、非单调执行。
3. **下游效用** — 在线 RL、离线重标、检索重排、规划——**效用成功 ≠ 保真**。

## 工程实践

| 步骤 | 建议 |
|------|------|
| 选型提问 | 要在线监控、离线过滤，还是闭开环 RL？先定输出形态 |
| 数据 | 成功-only 时间标签不够；显式加入停滞、回退、失败 |
| 标定 | 冻结 VLM 分数先差分/归一再当 reward |
| 延迟 | 大 VLM 逐步查询难；可分层：轻量局部 + 关键时刻重模型 |
| 索引 | 跟 [Awesome-Progress-Models](../../sources/repos/awesome-progress-models.md) 同步新论文 |

## 局限与风险

- 弱时间假设在重试/平台期失效。
- 纯 RGB 看不见力、滑移、已完成子目标计数（与 [力觉记忆](../entities/paper-fm-vla.md) 等正交）。
- 自动标注继承教师模型偏差；可被策略利用。
- 长程需要显式任务记忆，短窗不够。

## 关联页面

- [Progress Reward Survey（论文实体）](../entities/paper-progress-reward-modeling-survey.md) — 综述与 Awesome 入口
- [TOPReward](../entities/paper-topreward.md) — 视频 VLM token 似然零样本进度；OXE / ManiRewardBench
- [PRM-as-a-Judge](../entities/paper-prm-as-a-judge.md) — 冻结 PRM 打进度曲线，用 OPD 评 VLA/WAM 过程（非训练奖励）
- [Reinforcement Learning](../methods/reinforcement-learning.md) — 稠密奖励与信用分配
- [Imitation Learning](../methods/imitation-learning.md) — 演示时间序作弱进度
- [VLA](../methods/vla.md) — 指令微调进度模型常挂 VLM 生态
- [ACE-Brain-0.5](../entities/paper-ace-brain-0-5.md) — 进度估计内置于统一具身脑；RBM refined VOC 强
- [Contact-Rich Manipulation](./contact-rich-manipulation.md) — 接触细进度常需力/触觉
- [具身评测基准选型](../queries/embodied-eval-benchmark-selection-loop.md) — 效用评测语境
- [具身大模型分类学选型闭环](../queries/embodied-fm-taxonomy-loop.md) — 奖励/评测侧与 VLA·WM 选型并列阅读

## 参考来源

- [综述论文归档](../../sources/papers/progress_reward_modeling_survey_arxiv_2607_21655.md)
- [Awesome-Progress-Models](../../sources/repos/awesome-progress-models.md)
- [TOPReward 论文归档](../../sources/papers/topreward_arxiv_2602_19313.md)
- [PRM-as-a-Judge 论文归档](../../sources/papers/prm_as_a_judge_arxiv_2608_14284.md)

## 推荐继续阅读

- [arXiv:2607.21655](https://arxiv.org/abs/2607.21655) — 综述全文
- [Awesome-Progress-Models](https://github.com/sterzhang/Awesome-Progress-Models) — 可点击论文画廊
- [TOPReward 项目页](https://topreward.github.io/webpage/) — 冻结 VLM token 似然进度实例
- [PRM-as-a-Judge 项目页](https://prm-as-a-judge.github.io/) — 进度曲线作过程评测套件
