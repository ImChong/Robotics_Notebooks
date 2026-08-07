# AGILE 文档站（WBC-AGILE）

> 来源归档（site / project page）

- **标题：** AGILE — A Generic Isaac-Lab based Engine
- **类型：** documentation / project site
- **URL：** <https://nvidia-isaac.github.io/WBC-AGILE/>
- **代码：** <https://github.com/nvidia-isaac/WBC-AGILE>
- **论文：** <https://arxiv.org/abs/2603.20147>
- **机构：** 英伟达（NVIDIA）
- **核查日期：** 2026-08-07
- **一句话说明：** AGILE 官方文档站：定位为 Isaac Lab 外部项目，展示从任务设计、训练、评测到 Sim2Sim / 真机部署的完整 RL 管线，并列出多机器人任务 ID 与 Quick Start。

## 核心摘录（归纳，非全文）

- 首页强调：AGILE 是 **external Isaac Lab project**，覆盖 task design → training → evaluation → sim-to-real deployment。
- Key Features：多机器人（Booster T1 / Unitree G1）、Teacher–Student 蒸馏、单文件任务配置（共享 `agile/rl_env/mdp/`）、随机+确定性评测与 HTML/W&B、Sim-to-MuJoCo、OSMO 远程训练。
- Supported Tasks 表给出具体 Gym 风格 task ID（locomotion / height / stand-up / pick&place / whole-body tracking / debug GUI）。
- Quick Start 入口为 `scripts/train.py` 与 `scripts/eval.py`。

## 开源状态

- 文档站明确链接 GitHub；**已开源**（详见 [`sources/repos/wbc_agile.md`](../repos/wbc_agile.md)）。

## 对 wiki 的映射

- [AGILE（论文实体）](../../wiki/entities/paper-agile-humanoid-loco-manipulation.md)
- [Isaac Lab](../../wiki/entities/isaac-lab.md)

## 参考来源（原始）

- 文档站：<https://nvidia-isaac.github.io/WBC-AGILE/>
- GitHub：<https://github.com/nvidia-isaac/WBC-AGILE>
- arXiv：<https://arxiv.org/abs/2603.20147>
