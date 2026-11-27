# HappyWaterXP/HiMe — HiMe 官方开源仓库档案

- **GitHub 地址：** https://github.com/HappyWaterXP/HiMe
- **默认分支：** `main`（快照记录 31 commits、13 stars、2 forks）
- **开源协议：** 未声明（仓库元数据未标注协议，根目录文件清单无 `LICENSE` 文件）
- **关联论文：** *HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control*（[arXiv:2607.03449](https://arxiv.org/abs/2607.03449)）
- **核对边界：** 静态代码核对，未本地运行（基于仓库主页、`README.md` 与 `files_sample` 文件清单核查；仓库开源了任务服务端、Sentry/Planner 智能体调度、跨模态记忆模块、提示词模板及 WidowX-250 参考推理客户端，但不包含微调后的 $\pi_{0.5}$ 低层策略权重与 60 条真机示教数据，运行前需自行配置 VLM / Embedding API 端点及底层策略服务）。

## 核心入口与目录结构（已核实）

- **任务服务端与 HTTP API（`server/`）：**
  - `server/app.py`：FastAPI 服务入口（启动命令 `uv run uvicorn server.app:app --host 0.0.0.0 --port <PORT>`），暴露 `POST /tasks`（初始化全局指令、工作记忆窗口 `observer_window_size`、人为干预开关与初始双视角图像）、`POST /tasks/{task_id}/step`（上传主视角与腕部图像序列并驱动 Sentry/Planner 更新）与 `POST /tasks/{task_id}/user_instruction`（在线注入新用户指令 `user_new_instruction`），统一返回 `TaskPublicState`（含 `task_id`、`is_done`、`runtime_state`、`planner_status`、`plan_list`、`summary`、`current_subtask_description`）。
  - `server/task_manager.py`、`server/task_state.py`、`server/schema.py`、`server/config.py`、`server/model_config.py`、`server/image_utils.py`、`server/round_logger.py`、`server/ablation.py`：任务生命周期管理、状态机、图像拼接与消融开关控制。
- **Sentry 与 Planner 智能体及 VLM 客户端（`agent/`、`client/`）：**
  - `agent/observer.py`、`agent/multitag_planner.py`：分别实现基于滑窗观测的子目标完成判定（Sentry/Observer）与支持两轮检索-CRUD 的多标签规划器（Planner）。
  - `client/base_vlm_client.py`、`client/observer_vlm.py`、`client/planner_vlm.py`：封装面向 OpenAI API 或本地 vLLM OpenAI 兼容接口的调用客户端。
- **跨模态记忆与辅助脚本（`memory/` 及根目录）：**
  - `memory/encoder.py`、`memory/multitag_recorder.py`、`extractor.py`、`prompt_loader.py`、`main.py`：文本查询嵌入编码、跨模态多标签记录管理、XML 响应解析及提示词加载器。
- **任务与消融提示词（`prompt/`）：**
  - `prompt/README.md`、`prompt/task1.txt` / `task1_obs.txt`、`task2.txt` / `task2_obs.txt`、`task3.txt` / `task3_obs.txt`，以及针对任务 3（Rearrangement）的消融提示词：`task3_FIFO.txt`、`task3_hime_wo_sentry.txt`、`task3_no_management.txt`、`task3_only_image.txt`、`task3_only_text.txt`、`task3_transient_memory.txt`、`task3_transient_memory_wo_sentry.txt`。
- **真机参考客户端与基线服务（`widow/`、`flat_memory/`）：**
  - `widow/infer.py` & `widow/README.md`：WidowX-250 参考推理客户端（启动命令 `uv run python widow/infer.py --task_server_base_url <TASK_SERVER_BASE_URL> --policy_host <POLICY_HOST> --policy_port <POLICY_PORT> --policy_trace_npz_folder ./_policy_traces`），负责采集相机帧、与低层 VLA 策略通信并向任务服务端同步状态。
  - `flat_memory/app.py`、`flat_memory/app_no_memory.py`、`flat_memory/flat_memory_client.py`、`flat_memory/README.md`：独立的 Flat Memory / No Memory 基线服务（`uv run uvicorn flat_memory.app:app --host 0.0.0.0 --port <PORT>`）。
- **环境配置：**
  - `pyproject.toml`（要求 Python 3.13+，推荐 `uv venv && uv sync`）、`env.example.sh`（配置 planner / observer / embedding 端点、API Key、`PLANNER_PROMPT_NAME`、`OBSERVER_PROMPT_NAME` 及消融运行开关）。

## 复现与运行依赖边界

- **依赖环境与外部服务：** 需要 Python 3.13+ 环境，以及可用的 Planner VLM 端点（如 GPT-4o API 或本地 vLLM 部署的 Qwen3-VL-30B）、Sentry VLM 端点（如本地 vLLM 部署的 Qwen3-VL-8B）和文本 Embedding 端点（如 OpenAI `text-embedding-3`）。
- **低层控制策略解耦：** 仓库未内置 $\pi_{0.5}$ 模型权重或 OpenPI 训练脚本；`widow/infer.py` 通过 `--policy_host` / `--policy_port` 连接外部策略推理服务，真实部署需按具体硬件相机驱动、控制循环与策略通信协议进行适配。

## 与 wiki 的映射

- 对应实体页：[HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control](../../wiki/entities/paper-agentic-2607-03449.md)
