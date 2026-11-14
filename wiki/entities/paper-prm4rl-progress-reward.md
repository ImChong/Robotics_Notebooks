---
type: entity
project_id: prm4rl-progress-reward
tags: [paper, robotics, ingest-2026-11]
status: complete
updated: 2026-11-14
arxiv: "2610.22056"
code: https://github.com/deng-ai-lab/PRM4RL
related:
  - ./paper-coda-coevolving-data-policy.md
  - ./paper-spear-selective-preference.md
original:
  - ../../sources/papers/prm4rl-progress-reward_arxiv_2610_22056.md
sources:
  - ../../sources/papers/prm4rl-progress-reward_arxiv_2610_22056.md
  - ../../sources/repos/prm4rl-progress-reward.md
summary: "子任务进度函数驱动的稠密强化学习奖励"
---

# PRM4RL：大语言模型生成的任务进度奖励

## 一句话定义

**LLM 将稀疏成功任务分解为子任务并生成进度判定函数，再以势函数形式向 RL 提供低层稠密学习信号。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PRM | Progress Reward Model | 进度奖励模型 |
| LLM | Large Language Model | 大型语言模型 |
| RL | Reinforcement Learning | 强化学习 |
| PPO | Proximal Policy Optimization | 近端策略优化 |

## 为什么重要

子任务进度函数驱动的稠密强化学习奖励，关注现有控制、数据或机器人学习方法在复杂任务中的失效点。这里单独建论文实体，避免与其他论文共用详情页；已通读原始论文 HTML 的可用正文和附录；源码执行效果仍需复现。

## 核心机制

LLM 生成任务分解与子任务判定 Ψ、进度函数 Φ，策略从当前状态获取子任务先验，同时使用 r=γΦ(s_next)-Φ(s_now)+成功奖励 进行潜势塑形训练；论文使用 PPO、SAC 进行验证。

## 方法流程

该流程依据原文方法描述，不等价于源码调用时序。

```mermaid
flowchart LR
  task[自然语言任务和环境] --> llm[LLM 子任务与进度函数生成]
  llm --> prior[激活子任务先验]
  llm --> potential[进度势函数]
  potential --> shaping[稠密奖励塑形]
  prior --> rl[RL 策略优化]
  shaping --> rl
  rl --> env[仿真交互]
  env --> potential
```

## 源码运行时序图

依据官方仓库 README 与实际执行入口静态核查，尚未部署运行：

```mermaid
sequenceDiagram
    autonumber
    participant User as Runner
    participant Batch as run_mw.sh
    participant Train as train_mw.py
    participant Wrap as ContinuousTaskWrapper
    participant Func as progress_functions
    participant RL as SAC
    participant Env as MetaWorld
    User->>Batch: launch training batch
    Batch->>Train: task algorithm and seed
    Train->>Wrap: initialize env wrapper
    Wrap->>Func: evaluate progress
    Func-->>Wrap: potential and stage
    Wrap-->>RL: reward shaped observation
    RL->>Env: policy action
    Env-->>Wrap: next observation
    RL-->>Train: checkpoints and logs
```

## 实验与评测

在 MetaWorld 与 ManiSkill 仿真任务中以 5 个随机种子比较 Text2Reward、ELLM 和 Oracle 奖励；在论文所设环境中优于多种 LLM 规划和奖励基线。NeurIPS 2025 会议论文、2026 年 arXiv 编号不是同一发布日期。

## 局限与工程实践

- 势函数塑形的最优策略不变性需满足理论假设；仿真收益不代表任意真机或 RL 算法保证收敛。代码 README 提醒原实验 GPT-4o 不易精确复现。
- **源码状态：** 官方训练代码公开，README 包含 train_mw.py、train_ms.py 入口与 LLM 版本复现说明。
- **原文状态：** 已通读 arXiv HTML 的可用正文和附录；具体复现及跑通情况尚未独立验证。

## 结论

**将进度信号以势函数方式加入基础 RL，比直接让 LLM 写一套任意奖励更便于分析理论边界。**

1. 使用与论文一致的任务、硬件和评测协议复验，不跨实验口径比较数据。
2. 把方法改善的指标与额外的数据、计算或控制成本分开记录。
3. 对未经独立核查的复现入口，暂不声称完成验证。

## 与相关方法对比

| 对照方法 | 关键区别 | 局限或适用条件 |
|---|---|---|
| 稀疏成功奖励 | 长时任务中间反馈较少 | PRM 生成子任务进度函数作为稠密塑形信号 |
| 直接让 LLM 任意编写奖励 | 难保证最优策略不变等理论边界 | PRM 利用势函数差分，但保证依赖前提 |

不同基准和硬件的结果不能直接排序。

## 关联页面

- [CoDA：失败驱动的数据引擎与动作策略协同进化](./paper-coda-coevolving-data-policy.md)
- [SPEAR：机器人强化学习中的选择性 VLM 偏好询问](./paper-spear-selective-preference.md)

## 参考来源

- [论文归档](../../sources/papers/prm4rl-progress-reward_arxiv_2610_22056.md)
- [官方仓库归档](../../sources/repos/prm4rl-progress-reward.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2610.22056)
- [公开仓库](https://github.com/deng-ai-lab/PRM4RL)
