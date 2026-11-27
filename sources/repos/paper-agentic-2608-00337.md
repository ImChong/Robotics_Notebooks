# GaTech-RL2/armory — Armory 官方开源仓库档案

- **GitHub 地址：** https://github.com/GaTech-RL2/armory
- **默认分支：** `main`
- **开源协议：** Apache-2.0
- **关联论文：** *Action Chunk Scheduling for Batched Robot Policy Serving*（[arXiv:2608.00337](https://arxiv.org/abs/2608.00337)）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 核查，未在本地 GPU/Modal 环境启动服务器或连接机械臂/LIBERO 仿真执行）。

## 核心入口与目录结构（已核实）

- **本地 / 集群 / Modal 运行入口与配置：**
  - `scripts/serve.py`、`scripts/run.py`：本地或自托管部署的主入口，分别启动多机器人策略推理服务器与客户端机群（支持 `PI05`、`LIBERO` 及无 GPU 的 `policy:mock` 仿真模式）。
  - `scripts/modal/`（`app.py`、`images.py`、`run.py`、`serve.py`、`sweep.py`、`utils.py`）与 `scripts/sbatch/`（`launch_sweep.py`、`run_case.sh`、`collect_results.py`）：分别面向 Modal 云容器与 Slurm 集群的批实验与参数扫描入口。
  - `configs/`：`modal_server_lookahead.json`、`modal_single_robot_libero.json`、`modal_5_robots_libero.json`、`modal_10_robots_libero.json`、`modal_20_robots_libero.json` 及 `configs/server/`（`libero.json`、`libero_groot.json`、`mock.json`）。
- **服务端核心与多进程架构（`src/armory/serving/`）：**
  - `server.py`、`server_runtime.py`、`routes.py`、`protocol.py`、`session.py`、`slots.py`：WebSocket 前端服务、会话协议与共享内存槽位管理。
  - `scheduler.py`、`engine.py`、`config.py`、`schemas.py`、`rtc.py`：调度器主循环、GPU 推理引擎工作进程及实时分块相关配置。
- **调度算法与状态镜像（`src/armory/scheduling/`）：**
  - `base.py`、`baselines.py`（Round Robin 与 EDF 基线）、`lookahead_actions.py`（Lookahead 调度器）、`dynamic_action.py`、`mirror.py`（服务端动作队列与执行状态软件镜像）、`latency.py`（推理与网络通信延迟估计）。
- **策略后端适配层（`src/armory/backends/` 与 `src/backends/`）：**
  - `src/armory/backends/`：`registry.py`、`types.py`、`mock.py`、`inference_profiles.json`。
  - `src/backends/openpi_adapter/`（`policy_adapter.py`、`serve_factory.py`）与 `src/backends/gr00t_adapter/`（`policy_adapter.py`、`serve_factory.py`）：分别封装 OpenPI（$\pi_{0.5}$）与 NVIDIA Isaac GR00T 策略推理后端。
- **轻量机器人客户端（`armory-client/`）：**
  - `armory-client/src/armory_client/`：`client.py`、`action_chunk_broker.py`、`messages.py`、`msgpack_numpy.py`、`schemas.py`，供单台机器人端异步接收与消费动作分块。
- **评测与可视化（`src/evaluation/`、`scripts/visualization/`、`tests/`）：**
  - `src/evaluation/`：`runtime.py`、`envs/libero.py`、`envs/mock.py`、`agents/policy_agent.py`、`toxiproxy.py`（网络延迟/抖动注入）、`metrics/`（`loading.py`、`plots.py`、`server.py`、`summary.py`）。
  - `scripts/visualization/`：`action_fate.py`、`plot_sweep.py`。
  - `tests/`：`scheduling/mirror_test.py`、`scheduling/action_chunk_broker_test.py`、`serving/scheduler_worker_test.py`、`serving/server_smoke_test.py`、`backends/openpi_adapter_test.py`、`backends/gr00t_adapter_test.py`。

## 复现与运行依赖边界

- **运行时与包管理：** 需 Python 3.11 与 `uv`；第三方依赖通过 Git submodules（`third_party/` 下固定版本的 OpenPI、LIBERO 与 Isaac GR00T）及 `GIT_LFS_SKIP_SMUDGE=1 uv sync`（额外扩展 `--extra evaluation --extra serving-web`）安装。
- **硬件与权重边界：** 实机与全量策略评测需远端 GPU（论文主实验为单卡 NVIDIA L40S，可选 H100）及 $\pi_{0.5}$ / GR00T 检查点；无 GPU 环境下可使用 `policy:mock` 模式验证调度与通信链路。

## 与 wiki 的映射

- 对应实体页：[Action Chunk Scheduling for Batched Robot Policy Serving](../../wiki/entities/paper-agentic-2608-00337.md)
