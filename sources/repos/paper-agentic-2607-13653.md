# REAL 官方仓库（Sage-PhysicalAI/REAL）

- **官方仓库：** [Sage-PhysicalAI/REAL](https://github.com/Sage-PhysicalAI/REAL)（论文与 README 克隆地址为 [InternRobotics/REAL](https://github.com/InternRobotics/REAL)，项目主页为 [internrobotics.github.io/REAL](https://internrobotics.github.io/REAL/)）
- **关联论文：** [paper-agentic-2607-13653](../../wiki/entities/paper-agentic-2607-13653.md)（arXiv:2607.13653v2，ECCV 2026）
- **协议与状态：** MIT License，默认分支 `main`（60 stars，在线 RL 运行时位于独立 `gspo` 分支）；**部分开源**（代码、241 条 REAL-Bench 任务 YAML 与校验器、MCP 服务端、Agent 客户端、程序化数据生成与标注脚本、SFT/GSPO 训练脚本已公开；场景资产、MesaTask-10K USD 物体资产、录制轨迹、训练语料及模型权重未随仓分发）。
- **核对范围：** 基于仓库缓存快照（`context.json` `files_sample` 与 `readme_preview`、`project_page.txt`）完成静态结构与入口核对；**静态代码核对，未本地运行**。

## 仓库结构与已核实入口

| 路径 | 职责与已核实入口 |
|---|---|
| `benchmark/` & `real_bench/` | 241 条 REAL-Bench YAML 任务定义（`benchmark/tasks/{FDP,FODP,FDO,SUL}/`：`FDP=72, FODP=56, FDO=48, SUL=65`）、287 个 MesaTask USD 文件锁（`benchmark/mesa_required.txt`）及校验入口 `python -m real_bench`（`load_real_bench()`）。 |
| `mcp_server/` & `configs/` | MCP 服务端、工具注册、感知辅助与任务配置加载器 `mcp_server.config.load_task_config`；`configs/demo_task.yaml` 为可移植示例配置。 |
| `scripts/demo/run_mcp_server_demo.sh` | 在 InternUtopia + Isaac Sim 4.5.0（Python 3.10.15、`mcp==1.9.4`、`httpx==0.28.1`）环境下启动 MCP SSE 服务（默认 `http://127.0.0.1:8080/sse`，支持 `DEMO_TASK_CONFIG` 指定基准测试集单回合 YAML）。 |
| `agents/` | 两个 MCP 客户端参考实现：`agents/qwen_agent.py`（`python -m agents.qwen_agent` 本地 Qwen/LoRA 推理）与 `agents/vlm_api_agent.py`（`python -m agents.vlm_api_agent` OpenAI 兼容函数调用 API），共享 `agents/common.py`。 |
| `proc_datagen/` & `scripts/filter/` | `proc_datagen/task_generator.py`（任务生成与静态占位检测 `--verify-placement`）、`proc_datagen/verify_proc.py` 与 `scripts/filter/batch_filter_proc.sh`（Isaac Sim 物理稳定性筛选）、`proc_datagen/trajectory_annotation/annotate_trajectory.py`（离线轨迹分步标注）。 |
| `training/qwen3vl_sft/` | `training/qwen3vl_sft/train_qwen3vl_sft.sh`（接入上游 `QwenLM/Qwen3-VL` 的 `qwen-vl-finetune` 做 SFT 微调）。 |
| `training/mcp_gspo/`（`gspo` 分支） | 基于 `ms-swift` 与外部 MCP worker 的在线 RL 运行时（`git fetch origin gspo && git switch gspo`）。 |
| `tests/` | 无凭证回归测试：`tests/test_demo_launcher.py`、`tests/test_mcp_regressions.py`、`tests/test_agents_mcp_e2e.py`（通过真实 HTTP/SSE MCP 服务与调试用 World Graph 端到端校验 Agent 协议）。 |

## 论文与开源代码接口对照

- 论文正文及 Appendix C.2 中工具名为 `nav_to`、`walk_around`、`show_object_by_category`、`show_receptacles`、`gaze_at`、`pick`、`place`、`open`、`close`、`ask`、`finish`。
- 开源仓库 `README.md` 公布的 MCP 服务端实际暴露工具名为：`list_receptacles`、`navigate_to`、`explore_receptacle`、`focus_on`、`find_objects`、`highlight_receptacles`、`pick`、`place`、`open` / `close`、`finish`、`ask`。

## 复现边界声明

- 静态代码核对，未本地运行。
- 完整仿真评测与训练除本仓库外，需额外安装 InternUtopia 与 Isaac Sim 4.5.0、准备场景/元数据资产并单独下载 [MesaTask-10K](https://huggingface.co/datasets/InternRobotics/MesaTask-10K) 物体 USD（设置 `MESATASK_USD_ROOT`）；仓库不提供预训练 SFT/RL 检查点与真机低层 $\pi_{0.5}$ 权重。
