# zhangzhongbo2213/VABench（VA-Bench 官方代码仓库）

- **官方仓库：** https://github.com/zhangzhongbo2213/VABench
- **对应论文与实体：** [VABench: Measuring Embodied Spatial Intelligence through Visual Demonstrations, Active Perception, and Metric Control](../../wiki/entities/paper-agentic-2609-19554.md)（`arXiv:2609.19554`）
- **开源协议与分支：** MIT License，默认分支 `main`（截至采集快照：15 stars，15 commits）
- **资产与数据集分发：** [GitHub Releases (`assets-v1`)](https://github.com/zhangzhongbo2213/VABench/releases/tag/assets-v1) 与 [Hugging Face (`zhangzhongbo2213/VA-Bench`)](https://huggingface.co/datasets/zhangzhongbo2213/VA-Bench)
- **复现与核查边界：** 静态代码核对（基于 `context.json` 中的仓库文件列表、`README` 预览及论文附录 C/D 说明），未在本地运行仿真与评测脚本

## 目录结构与关键入口（已核对）

| 路径 / 模块 | 作用说明 | 来源依据 |
|---|---|---|
| `script/install.sh` | 创建并配置 `va-bench`（Python 3.10）及 RoboTwin 运行环境 | `README` 预览 |
| `script/run_benchmark.py` | 基准评测主入口，支持指定 `--run-id`、`--tasks`、`--num-seeds`、`--gpus`、`--parallel` 及 `--wire-api` | `README` 预览 |
| `script/monitor.py` | 监控多任务评测批次运行进度 | `README` 预览 |
| `configs/main_tasks.json` | 14 个主任务族及已验证种子清单配置 | `README` 预览 |
| `agent/src/agent/cli.py`、`client.py`、`config.py` | Agent CLI 与多后端 API 适配（支持 OpenAI `chat_completions` / Responses 与 `anthropic_messages` 协议） | `files_sample`、`README`、附录 D.1 |
| `agent/src/agent/harness.py`、`protocol.py`、`context.py`、`session.py` | 持久化决策闭环、JSON 协议解析、上下文裁剪压缩与 JSONL 轨迹记录 | `files_sample`、附录 D.1/D.4 |
| `agent/src/agent/robotwin/eval.py`、`loop.py`、`adapter.py`、`validator.py` | RoboTwin 评测循环、模型输出参数校验/范围裁剪、局部旋转补偿分解与环境执行适配 | `files_sample`、附录 D.1–D.2 |
| `agent/src/agent/robotwin/expert_learning.py`、`prompts.py`、`visuals.py` | RGB 示范帧采样检查门控、结构化文本摘要生成与视觉渲染处理 | `files_sample`、附录 D.2–D.3 |
| `agent/scripts/select_safe_eval_seeds.py` | 有序候选种子物理静置筛选与 `play_once()` 验证探针 | `files_sample`、附录 C.4 |
| `envs/_base_task.py`、`envs/_single_right_arm_place_target_task.py`、`task_config/demo_clean.yml` | 确定性场景初始化、`demo_clean` 评测配置与共享放置成功谓词 | `project_page.txt`、附录 C.4 |
| `active_spatial_benchmark_xyz/active_spatial_benchmark/` | 主动感知与抓取/视角辅助模块（`env.py`、`actions.py`、`camera_control.py`、`camera_layout.py`、`trajectory_guard.py`、`active_belief.py` 等） | `files_sample` |

## 典型运行与协议要点

- **环境安装与启动**：通过 `conda create -n va-bench python=3.10 -y` 后执行 `bash script/install.sh`；设置 `AGENT_BASE_URL`、`AGENT_MODEL`、`AGENT_API_KEY` 环境变量（可选 `AGENT_WIRE_API=anthropic_messages`）后调用 `python script/run_benchmark.py` 启动单任务冒烟或全量 14 任务评测。
- **单线隔离契约**：模型与仿真环境之间通过 `harness.py` / `validator.py` / `robotwin/adapter.py` 严格解耦；JSONL 日志与干净 RGB 图像分离保存，调试/标注视图不进入模型上下文。
