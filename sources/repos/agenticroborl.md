# AgenticRoboRL 官方代码仓库

> 来源归档（repo）

- **标题：** AgenticRoboRL: Bringing Coding-Agent Post-Training to Agentic Robotics
- **类型：** repo（机器人 planner 的 SFT + multi-turn RL 后训练框架）
- **链接：** <https://github.com/xinxinliu-ai/AgenticRoboRL>
- **论文：** <https://arxiv.org/abs/2610.28589>（见 [论文来源归档](../papers/agenticroborl_arxiv_2610_28589.md)）
- **项目页：** <https://xinxinliu-ai.github.io/AgenticRoboRL/>（见 [项目页归档](../sites/agenticroborl-project.md)）
- **模型：** <https://huggingface.co/ciciciciliu/AgenticRoboRL-Qwen3.5-9B>
- **许可证：** Apache-2.0
- **入库日期：** 2026-10-26
- **一句话说明：** 在保留机器人 harness 执行组件的前提下，记录 planner 的真实模型请求，将 episode 结果映射为可复用的训练样本，并以 SFT 和多轮 RL 更新 planner。
- **沉淀到 wiki：** [wiki/entities/paper-agentic-roborl.md](../../wiki/entities/paper-agentic-roborl.md)

## 仓库结构与运行入口（依据 README）

| 路径 / 入口 | 用途 |
|---|---|
| rl/proxy.py | 代理 harness 与 vLLM model server，记录上下文、生成 token 与 rollout log-probability |
| rl/build_round.py、rl/adapters/ | 将 episode 结果与模型调用关联，生成每轮 RL 样本 |
| rl/train_rl.py | dual-clip PPO planner 更新；训练前检查 prompt token 是否重现 |
| rl/local_ctl.py、rl/multi_node_ctl.py | 单机或多机 rollout、训练和评测编排 |
| sft/rpent/、sft/capx/ | 两种 harness 的 teacher 轨迹处理与 SFT 样本构造 |
| harnesses/minimal/ | 无 GPU / simulator / network 的最小端到端训练回路 |
| harnesses/rpent_libero/、harnesses/capx_libero/ | 论文两种实验 harness 配置 |

## 复现条件与开源边界

- README 列出的训练环境使用 Python 3.11、PyTorch 2.13、Transformers 5.17、vLLM 0.28；具体版本以对应 environment 为准。
- RPent 完整复现依赖 RPent 指定 checkout、LIBERO-PRO、冻结的 Pi0.5 checkpoint、SAM3 checkpoint 及 model/training 环境；CaP-X 需要其 LIBERO 环境。
- 仓库提供无需 GPU、模拟器或网络的 minimal harness quick start，适合先验证控制器与 RL 记录/训练流程。
- README 明确 teacher SFT 轨迹未发布；需使用已发布 SFT checkpoint 启动 RL，或自行准备监督数据。
- Hugging Face 提供 RPent 的 Qwen3.5-9B SFT checkpoint 与第 6 轮 RL checkpoint；checkpoint 并不等同于教师训练集公开。

## 主要结果（论文 / README）

- **RPent / LIBERO-PRO：** 18.1% → 65.1%（SFT）→ 71.0%（SFT + RL），800 个评测 episodes。
- **RPent / LIBERO-Plus：** 19.2% → 66.7% → 72.2%，使用相同 checkpoint 且无额外训练。
- **CaP-X / LIBERO-PRO：** 2.7% → 18.7% → 28.5%；该 harness 生成控制代码，实验设置包含 privileged state。
