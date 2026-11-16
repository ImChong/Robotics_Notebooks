# Gödel Agent 官方源码归档

- **类型：** repo
- **项目：** Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement / Gödel Agent：用于递归自我改进的自指智能体框架
- **代码：** https://github.com/Arvid-pku/Godel_Agent
- **源码快照：** https://github.com/Arvid-pku/Godel_Agent/tree/bbb508796be31c7140cdfc7106efd830a1324242
- **论文：** [arXiv:2410.04444v4](../papers/godel_agent_arxiv_2410_04444v4.md)
- **许可：** MIT（官方仓库 LICENSE）
- **开放状态：** 已开源；有 `python src/main.py` 执行入口与四类任务评价模块，但必须自备 API 密钥与数据/环境。未在本次 ingest 实际运行。
- **核对日期：** 2026-11-08
- **核对范围：** 官方 README 的安装/配置与结构；`src/main.py`、`src/agent_module.py` 的 Agent/动作/递归逻辑、`src/logic.py` 的源代码检查与导出、`src/goal_prompt.md`、`src/task_drop.py` 的评测入口及 `src/` 四任务模块目录、requirements。未全量执行任务评测。
- **通读：** 已完成

## 官方运行入口与真实调用链

1. README：`pip install -r requirements.txt`；将 OpenAI API key 配到 `src/key.env`；在根目录运行 `python src/main.py`。
2. `main.py`：`Agent(goal_prompt_path="src/goal_prompt.md", key_path=key_path)` → `reinit()` → `evolve()`。
3. `Agent.__init__`：读取 goal prompt、key，设置 `goal_task=task_mgsm.MGSM_Task()`（默认 MGSM），注册 introspection/logic modification/execute/LLM/evaluation 工具。
4. `reinit()`：`action_environment_aware` 和 `action_read_logic("agent_module", "solver")` 将初始状态和 solver 源码送入 history。
5. `evolve()`：使用配置的模型（当前源码为 `o3-2025-04-16`）与工具列表产生 tool calls → `execute_action()`；尾部递归再调用 `evolve()`。
6. `action_adjust_logic`：`importlib.import_module`、`exec(compile(...))`、`setattr` 动态更新函数/方法；`__source__` 保留新定义供后续读取。
7. `action_evaluate_on_task`：调用 `task.evaluate(solver)`（默认 MGSM）获得反馈；若超过 `task_mgsm.last_test_acc`，用 `logic.store_all_logic()` 导出代码快照。

## 代码与论文差异及风险

- README 所述源码可运行入口已经公开，但不是“无需凭证即可复现”；代码提供 `key.env` 空文件（不要提交凭证）。
- **版本差异：** 论文的基准 GPT-3.5 / 优化器 GPT-4o，与当前 `agent_module.py` 中优化器 `o3-2025-04-16`、初始 solver `gpt-4o-mini-2024-07-18` 不一致。按 README 默认运行**不能直接验证论文原始 Table 1**。
- `action_run_code` 包含执行 Python/Bash，`action_adjust_logic` 可执行动态代码；官方 goal prompt 授予广泛权限。应在隔离、限权、网络受控环境复现，明确成本、数据集、预算和停止条件。
- 本次仅静态阅读，没有跑 Python 任务、调用付费模型，也未声明复现其分数。

## 对 wiki 的映射

- [Gödel Agent 实体](../../wiki/entities/paper-godel-agent.md)
- [论文来源](../papers/godel_agent_arxiv_2410_04444v4.md)

## 一手来源

- https://github.com/Arvid-pku/Godel_Agent/blob/bbb508796be31c7140cdfc7106efd830a1324242/README.md
- https://github.com/Arvid-pku/Godel_Agent/blob/bbb508796be31c7140cdfc7106efd830a1324242/src/main.py
- https://github.com/Arvid-pku/Godel_Agent/blob/bbb508796be31c7140cdfc7106efd830a1324242/src/agent_module.py
- https://github.com/Arvid-pku/Godel_Agent/blob/bbb508796be31c7140cdfc7106efd830a1324242/src/logic.py
- https://github.com/Arvid-pku/Godel_Agent/blob/bbb508796be31c7140cdfc7106efd830a1324242/src/goal_prompt.md
