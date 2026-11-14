# Progress Reward Model for Reinforcement Learning via Large Language Models — 官方代码仓库

- **官方仓库：** https://github.com/deng-ai-lab/PRM4RL
- **论文：** https://arxiv.org/abs/2610.22056
- **项目：** 论文官方原文
- **开放结论：** 官方训练代码公开，README 包含 train_mw.py、train_ms.py 入口与 LLM 版本复现说明。
- **核对范围：** 仓库首页 README、项目开放标识；真实源码运行入口和调用链仍待逐项核实
- **通读：** 未完成

## 研究对应关系

LLM 生成任务分解与子任务判定 Ψ、进度函数 Φ，策略从当前状态获取子任务先验，同时使用 r=γΦ(s_next)-Φ(s_now)+成功奖励 进行潜势塑形训练；论文使用 PPO、SAC 进行验证。

## 复现说明

当前未声明已实际运行。先按仓库 README 核对模型、依赖、数据和入口，再补运行时序图。

## 对 wiki 的映射

- [论文详情](../../wiki/entities/paper-prm4rl-progress-reward.md)
