# midea-ai/ars 官方代码核查

- **代码仓库：** <https://github.com/midea-ai/ars>
- **开源协议与分支：** Apache-2.0（`third_party/RL4VLA` 子模块保留其各自上游协议），默认分支 `main`
- **对应论文：** [ARS: Agentic Reward System for Robot Learning (arXiv:2609.34484v1)](../papers/paper-agentic-2609-34484.md)
- **核心依赖与环境边界：** Python 3.11；默认推理配置面向 `Qwen3.6-27B`，测试环境采用 `vLLM 0.19.1`、`PyTorch 2.10.0`、`Transformers 5.9.0`（在单卡 96 GiB VRAM GPU 上测试）；多节点扩展可选 Ray；语义错配基准采集依赖独立仿真环境中的 `third_party/RL4VLA` 子模块（与主推理环境隔离）。
- **核对范围：** 静态核对 GitHub 仓库元数据、`README.md` 预览（`project_page.txt`）与 `files_sample` 文件清单（覆盖 `agent/`、`configs/`、`examples/`、`scripts/` 与 `.gitmodules`）。静态代码核对，未本地运行。
- **通读：** 未完成
- **开源范围判断：** **已开源** ARS 完整推理智能体、跨智能体工作流与脚手架工具实现（`agent/`）、服务端与任务配置（`configs/`）、单视频匹配/错配示例与输出样张（`examples/`），以及语义错配基准的数据转换与评测脚本（`scripts/`）；同仓或页面提及的 `robometer/robometer` 与 `gen-robot/RL4VLA` 分别为外部基线对比仓库与仿真数据采集依赖子模块，非 ARS 本体。

## 已核对目录与入口

1. **CLI 主入口与批量/分布式推理（`agent/`）：**
   - `agent/cli.py`：提供 `run-batch`（面向含 `meta/info.json` 与 Parquet `image` 特征的 **LeRobot v2.1** 数据集）与 `run`（单视频目录调试）两种模式；支持 `--start-server configs/server.yaml` 自动拉起推理服务、`--proposal-batch-results` 接入外部进度提案审计以及 `--num-nodes` 启用 Ray 多节点分发。
   - `agent/serving.py` 与 `configs/server.yaml`：管理 vLLM 本地模型服务（需配置 `--allowed-local-media-path` 供服务端读取智能体生成的视觉裁剪图像/视频）。
   - `agent/batch_runner.py`、`agent/ray_batch_runner.py`、`agent/ray_worker.py`、`agent/batch_journal.py`、`agent/result_index.py`：批量任务调度、断点日志与 `batch_results.jsonl` 清单索引。
2. **智能体运行时、脚手架与工具协议（`agent/`）：**
   - `agent/runner.py`、`agent/session.py`、`agent/client.py`、`agent/config.py`、`agent/workflows/default.yaml`：控制子智能体提案、主智能体独立验证与进度估计三阶段状态流转。
   - `agent/tool_protocol.py`、`agent/schemas.py`、`agent/diagnostics.py`、`agent/visual.py`、`agent/progress.py`、`agent/is_success.py`、`agent/task_context.py`、`agent/reference_examples.py`：实现 `inspect_visual` 视觉裁剪与抽帧、事件时间线覆盖与重验证约束、$[0, 1000]$ 进度区间编辑/校准与结构校验、终局标签与图文参考注入。
   - `agent/lerobot_source.py`、`agent/episode_source.py`：LeRobot v2.1 与单视频轨迹读取适配。
   - `agent/prompts/`：`common.md`、`reference_examples.md`、`trajectory_outcome_ground_truth.md` 及分阶段提示 `stages/bootstrap_observations.md`（事件提案）、`stages/challenge_observations.md`（独立挑战验证）、`stages/progress.md`（进度估计）。
3. **配置、示例与基准评测脚本（`configs/`、`examples/`、`scripts/`）：**
   - `configs/config.yaml`、`configs/server.yaml`、`configs/task_context.yaml`：配置智能体/服务端口、终局标签与参考例开关。
   - `examples/inputs/` 与 `examples/outputs/`：包含 `instruction_match` 与 `instruction_mismatch` 两组共享 `video.mp4`、含 Robometer 提案 `progress.csv` 及 ARS 审计输出的样例（`corrected_progress.csv`、`corrected_progress_plot.png`、`corrected_progress_review.mp4`、`final_summary.md`）。
   - `scripts/convert_rl4vla_mp_collect_to_lerobot.py`：将 `third_party/RL4VLA/ManiSkill` 采集的 `PutOnPlateInScene25MultiCarrot-v1` 轨迹转为含正确与错配双样本的 LeRobot v2.1 数据集。
   - `scripts/evaluate_progress_score.py` 与 `scripts/evaluate_progress_score.yaml`：计算语义错配基准的 **VOC**、**FPSR** 与 **CMPG** 指标。

## 边界说明

- 本归档基于公开仓库 `README.md` 与目录文件清单完成静态核对，**未在本地启动 vLLM `Qwen3.6-27B` 服务或实际运行 `agent.cli` 与 RL4VLA 仿真采集脚本**。
- `README.md` 指出默认配置在 96 GiB 显存 GPU 上测试，且下游 $\pi_{0.5}$ 策略微调与真实机械臂实验训练代码未包含在本仓库内（仓库聚焦 ARS 推理引擎与语义错配基准）。

## 映射

- [ARS 论文实体与源码运行时序图](../../wiki/entities/paper-agentic-2609-34484.md)
