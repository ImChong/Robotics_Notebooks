# MemCompiler（air-embodied-brain/MemCompiler）

- **URL：** <https://github.com/air-embodied-brain/MemCompiler>
- **组织：** Embodied Brain @ AIR, Tsinghua University
- **关联项目页：** [MemCompiler 项目页](../sites/memcompiler.md)
- **关联论文：** [MemCompiler arXiv 摘录](../papers/memcompiler_arxiv_2605_07594.md)

## 一句话说明

官方实现以 Manager–Executor 架构运行具身任务：Memory Compiler 根据逐步更新的 Brief State 编译相关记忆，Executor 负责环境动作；层次化记忆包括 Insight Graph、Query Graph 与 Interaction Graph。

## 开源状态（2026-10-11）

**代码已公开，提供运行与评测入口。** README 描述 ALFWorld 与 ScienceWorld 两套环境，运行脚本分别为 `run_alfworld.sh`、`run_scienceworld.sh`；对应 Python 入口为 `tasks/run_manager_executor.py`、`tasks/run_manager_executor_scienceworld.py`。训练模型评测入口为 `tasks/eval_with_trained_model.py`。

README 要求 Python 3.12、依赖安装、环境数据及路径设置，并配置云端 LLM API key 或本地 Manager / Executor 模型路径。项目页所列 Hugging Face 模型页暂时没有 model card；权重/数据覆盖范围和许可应单独核验。

## 运行入口

| 任务 | README 入口 |
|------|-------------|
| ALFWorld | `bash run_alfworld.sh` 或 `python tasks/run_manager_executor.py` |
| ScienceWorld | `bash run_scienceworld.sh` 或 `python tasks/run_manager_executor_scienceworld.py` |
| 已训练模型评测 | `python tasks/eval_with_trained_model.py` |

**复现边界：** 运行依赖目标环境及其数据、模型文件与 API 配置；论文报告的 SFT / GRPO 训练使用 8×NVIDIA H20，不能将代码公开等同于普通单机可完整复现实验。

## 交叉链接

- [MemCompiler 实体页](../../wiki/entities/paper-memcompiler-state-conditioned-memory.md)
- [项目页归档](../sites/memcompiler.md)
- [论文摘录](../papers/memcompiler_arxiv_2605_07594.md)
