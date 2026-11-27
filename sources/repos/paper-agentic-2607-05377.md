# Sage-PhysicalAI/Cortex — Cortex 官方开源仓库档案

- **GitHub 地址：** https://github.com/Sage-PhysicalAI/Cortex （论文与项目页外链 `https://github.com/steinate/Cortex` 对应此仓库）
- **默认分支：** `main`
- **开源协议：** 仓库含 `LICENSE` 文件（GitHub 元数据识别为 `NOASSERTION`）
- **关联论文：** *Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation*（[arXiv:2607.05377](https://arxiv.org/abs/2607.05377)）
- **关联模型与数据集：** Hugging Face 模型 [`Steinate/Cortex`](https://huggingface.co/Steinate/Cortex)、数据集 [`Steinate/Cortex`](https://huggingface.co/datasets/Steinate/Cortex)
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 核查；未在本地下载权重或启动 GPU 推理与标注服务）。

## 核心入口与目录结构（已核实）

- **System-2 推理与评测模块（已开源）：**
  - `scripts/run_scripts/step_level.sh` 与 `cortex/inference/step_level_eval.py`：帧级（Teacher-Forced）子任务与记忆预测评测入口，配合 `cortex/inference/config/sys2_subtask_val.json`、Base Model（`Qwen/Qwen3-VL-8B-Instruct`）与 Judge Model（`Qwen3.5-9B`）评估 `spatial`、`counting`、`long` 三类切片。
  - `scripts/run_scripts/episode_level.sh` 与 `cortex/inference/episode_level_eval.py`：回合级（Self-Forced）闭环自回归记忆评测入口，检测长程语义漂移（支持 `galaxea`、`agibot`、`behavior` 等数据集标签）。
  - `cortex/inference/eval_subtask_dataset.py`、`cortex/inference/subtask_eval.py`、`cortex/inference/eval_utils.py`：评测数据集加载、评测循环与 Judge 打分辅助工具；`annotations/` 下附带 `agibot_val_*.jsonl`、`behavior1k_val_*.jsonl`、`galaxea_val_*.jsonl` 验证集标注。
- **异步部署与 WebSocket 服务端（已开源）：**
  - `deployment/model_server/server_sys2.py`：本地 System-2 模型推理服务端（另附 `server_sys2_gemini.py` 与 `server_sys2_gpt.py` 对接云端视觉模型）。
  - `deployment/model_server/tools/websocket_policy_server.py`、`websocket_policy_client.py`、`image_tools.py`、`msgpack_numpy.py`：System-2 策略异步 WebSocket 通信与多视角图像/NumPy 消息序列化工具（`step_level.sh` 亦支持 `POLICY_BACKEND=websocket` 连接预启动服务）。
- **三阶段子任务自动标注与采样可视化（已开源）：**
  - `scripts/run_scripts/run_vlm_annotation.sh` → `cortex/annotation/vlm_annotation.py`：调用 OpenAI 兼容 VLM 接口生成种子子任务标签。
  - `scripts/run_scripts/run_manual_annotation_server.sh` → `cortex/annotation/annotator_server.py`：在本地 `http://127.0.0.1:8765` 启动浏览器人工纠错服务。
  - `scripts/run_scripts/run_dynamic_matching.sh` → `cortex/annotation/dynamic_matching.py`：基于 `observation.state`、`action` 及三视角视频（head / left-wrist / right-wrist）的可变模板动态规划传播。
  - `assets/dump_bin_bigbin/`：自带 10 条 LeRobot 格式示例轨迹（含 `parquet` 状态动作与三视角 `mp4` 视频）用于跑通标注演示；`visualize_subtask_sampling.py` 与 `scripts/run_scripts/run_subtask_visualization.sh` 用于子任务采样可视化。
- **数据配置与文档：**
  - `cortex/dataloader/qwenvl_llavajson/qwen_data_config.py`：Qwen-VL 数据加载配置；`docs/installation.md`、`docs/datasets.md`、`docs/annotation.md`；`Makefile`、`pyproject.toml`、`requirements.txt`。

## 复现与运行依赖边界

- **开源范围（部分开源）：** README TODO 明确勾选已发布 System-2 评测代码、子任务标注流水线与子任务标注数据集；**LIBERO / RoboTwin 的 System-1/2 闭环评测代码与 System-2 分布式训练代码尚在 TODO 待发布列表**。
- **外部依赖：** 运行 System-2 评测需下载 `Steinate/Cortex`、`Qwen/Qwen3-VL-8B-Instruct` 及裁判模型 `Qwen3.5-9B`，并配置 AgibotWorld / Galaxea / BEHAVIOR-1K 原始数据集；运行 VLM 初始标注需在环境变量中提供 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 与 `OPENAI_MODEL`。

## 与 wiki 的映射

- 对应实体页：[Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation](../../wiki/entities/paper-agentic-2607-05377.md)
