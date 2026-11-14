# Progress Reward Model for Reinforcement Learning via Large Language Models

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** Progress Reward Model for Reinforcement Learning via Large Language Models
- **中文：** PRM4RL：大语言模型生成的任务进度奖励
- **arXiv：** 2610.22056
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已直接阅读 arXiv HTML 摘要、方法与部分实验/局限章节；尚未逐行检查全文及全部附录，后续继续核对
- **通读：** 未完成
- **论文：** https://arxiv.org/abs/2610.22056
- **HTML：** https://arxiv.org/html/2610.22056
- **PDF：** https://arxiv.org/pdf/2610.22056
- **仓库：** https://github.com/deng-ai-lab/PRM4RL

## 论文机制

LLM 生成任务分解与子任务判定 Ψ、进度函数 Φ，策略从当前状态获取子任务先验，同时使用 r=γΦ(s_next)-Φ(s_now)+成功奖励 进行潜势塑形训练；论文使用 PPO、SAC 进行验证。

## 实验与证据边界

在 MetaWorld 与 ManiSkill 仿真任务中以 5 个随机种子比较 Text2Reward、ELLM 和 Oracle 奖励；在论文所设环境中优于多种 LLM 规划和奖励基线。NeurIPS 2025 会议论文、2026 年 arXiv 编号不是同一发布日期。

## 复现与开放状态

官方训练代码公开，README 包含 train_mw.py、train_ms.py 入口与 LLM 版本复现说明。

## 局限

势函数塑形的最优策略不变性需满足理论假设；仿真收益不代表任意真机或 RL 算法保证收敛。代码 README 提醒原实验 GPT-4o 不易精确复现。

## 对 wiki 的映射

- [PRM4RL：大语言模型生成的任务进度奖励](../../wiki/entities/paper-prm4rl-progress-reward.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.22056)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
