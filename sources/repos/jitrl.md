# JitRL 官方代码仓库（liushiliushi/JitRL）

> 来源归档（repo）

- **类型：** repo
- **地址：** <https://github.com/liushiliushi/JitRL>
- **许可证：** 仓库含 `Jericho/LICENSE`（Jericho 子目录）；根目录许可证以仓库为准，本次未逐一核对
- **对应论文：** [arXiv:2601.18510](https://arxiv.org/abs/2601.18510)（JitRL）
- **入库日期：** 2026-11-07
- **核对方式：** 浅克隆（`--depth 1`，HEAD `143d221`，提交日期 2026-07-21）；代码阅读，未运行实验（未配置 OpenRouter / WebArena 环境）
- **开源状态：** 已开源（训练无关的推理与评测代码；**无模型训练代码**，因方法本身不做梯度更新）

## 目录与入口（已读）

- `Jericho/main.py`：入口，解析参数后调用 `GameEvaluator.run_evaluation()`。
- `Jericho/src/evaluation.py`：`run_game_episode` 循环（`env.reset` → `agent.start_episode` → `agent.generate_action` → `env.step` → `agent.update_game_history_reward` → `agent.end_episode`）。
- `Jericho/src/jitrl_agent.py`：`generate_action`（候选动作生成与 logprob 提取）、`update_scores`（检索、优势估计、logit 修正）、`end_episode`。
- `Jericho/src/cross_episode_memory.py`：`retrieve_similar`（状态 + 历史双向量索引加权，默认 0.25 history + 0.75 state，另有 Jaccard 路径）、`add_episode`（写入经验，步级分数由 LLM 评估器给出）。
- `WebArena/run.py`：WebArena 评测入口（BrowserGym 环境，包装 `BrowserEnv.step`）。
- `WebArena/memory_agents/jitrl_agent.py`：WebArena 版 `BrowserGymJitRLAgent`，`update_scores` 中按归一化动作修正 logit。
- `WebArena/test_webarena_lite.py`：WebArena-Lite 批量测试入口。

## 与论文的对照（阅读所见，未运行复现）

- Jericho 默认超参（`--gamma 0.5`、`--exploration_rate 0.65`、`--llm_temperature 0.8`）与论文附录 H 表 10 一致。
- WebArena 入口 README 默认 `--max_steps 30`，论文附录 H 表 11 为 step limit 10：复现时以论文设定为准，二者不同。
- README 注明：论文 Gemini-2.5-Flash 结果使用的 OpenRouter 快照已下线，默认模型改为 `google/gemini-2.5-flash`，数字可能略有差异（仓库自述）。

## 对 wiki 的映射

- [paper-jitrl-just-in-time-rl](../../wiki/entities/paper-jitrl-just-in-time-rl.md)
- 源码运行时序图见该页「源码运行时序图」小节。
