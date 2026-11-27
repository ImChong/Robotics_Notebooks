# LLM Robot Cognition 官方代码库｜ShinasShaji/llm-robot-cognition

- **官方仓库：** <https://github.com/ShinasShaji/llm-robot-cognition>
- **项目主页：** <https://shinasshaji.github.io/icra-llm-robot-cognition/>
- **关联论文：** [From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?](https://arxiv.org/abs/2603.03148)（ICRA 2026）
- **分支 / 许可证 / 开源状态（2026-11-27 核查）：** 默认分支 `main`；仓库文件清单中未包含独立 `LICENSE` 文件（许可证未明确声明）。**已开源**：完整发布 PyBullet 两室一廊家居仿真世界、全向移动机械臂（KUKA LBR `iiwa` URDF 与网格资产）、8 项世界交互与记忆工具、ChromaDB 情景记忆模块、多后端 LLM 加载器、批量实验评测与结果绘图脚本。
- **核对范围：** 静态代码核对，未本地运行。直接核对了 `ShinasShaji/llm-robot-cognition` 默认分支 `main` 的根目录文件列表（含 `models/` 资产目录）、`README.md` 完整说明文档以及示例配置 `config.yaml`。

## 仓库结构与可运行入口（来自文件清单与 README 核对）

- **交互式仿真与批量评测入口：**
  - `simulation.py`：主交互运行入口。启动 PyBullet GUI 窗口，初始化房间环境与全向移动机械臂，并提供命令行交互提示以输入自然语言任务指令（如移动到厨房货架、抓取杯子等），驱动 LLM 工具调用闭环。
  - `evaluation.py`：批量自动化评测脚本。读取 `config.yaml` 中的仿真步长（`time_step`）、`headless` 模式、重复次数（`num_trials`）、记忆累积轮次（`num_memory_trials`，第 0 轮清空记忆、后续轮次复用并累积库内记录）、模型列表（`models`）、任务提示词（`prompts`）与成功判定条件（`conditions`），并将评测指标追加写入 `results.csv`。
  - `plot_results.py`：读取 `results.csv` 并在 `plots/` 目录生成性能与记忆对比图表。
- **环境、机器人与状态验证：**
  - `world.py`：构建 PyBullet 家居环境及语义导航拓扑图（通过 `Location` 与 `add_next_to` 连接厨房、客厅、走廊等节点），并在 `create_default_physical_objects()` 中加载桌面、货架及可交互物体网格（`models/Mug.obj`、`models/Shelf.obj`、`models/TV.obj`、`models/Table.obj`、`models/Walls.obj`）。
  - `robot.py`：封装由 `models/iiwa.urdf`（及 `models/meshes/link_0`–`link_7`、`finger_*`）构建的 7-DoF 机械臂与全向移动底盘控制。
  - `world_state_checker.py`：根据 `config.yaml` 中定义的 `occupied_by` 与 `swap` 等条件，核验物理仿真世界的真实终止状态（ground-truth success）。
- **智能体认知、记忆与工具层：**
  - `toolkit.py`：实现供 LLM 函数调用的 8 个工具（`Look around`、`Move to`、`Grab`、`Place`、`Search memory`、`Add to scratchpad`、`View scratchpad`、`End task`）。
  - `memory.py`：基于 ChromaDB 实现情景记忆向量存储与语义检索；在 `End task` 时写入任务描述、自判状态与执行摘要，并在后续任务通过 `Search memory` 检索。
  - `model_loader.py`：封装云端 API（OpenAI、Anthropic）与本地 OpenAI 兼容服务端（如 `vLLM` / `llama.cpp` server，配置 `LOCAL_API_HOST` 与 `MODEL_NAME`）的统一模型接口。
  - `counter.py`、`logging_utils.py`：工具调用计数与运行日志记录。
  - `.env.example`、`requirements.txt`、`architecture.md`：环境变量模板（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`LOCAL_API_KEY` 等）、Python 3.13 依赖声明与架构说明文档。

## 复现边界说明

- **静态代码核对，未本地运行：** 本次仅核对了仓库目录结构与 `README.md` 中的环境要求、运行命令及 `config.yaml` 示例，未在本地执行 `pip install -r requirements.txt` 或启动 PyBullet/LLM 服务复现实验。
- **运行前提与接口依赖：** 运行需 Python 3.13 环境以及可用的 OpenAI / Anthropic API Key 或本地托管的 OpenAI 兼容推理服务（`LOCAL_API_HOST`）。

## 对 wiki 的映射

- [From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?](../../wiki/entities/paper-agentic-2603-03148.md)
- [论文原文核查](../papers/paper-agentic-2603-03148.md)
