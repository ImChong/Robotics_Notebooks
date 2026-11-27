# NVlabs/VoLoAgent（与配套基准仓 NVlabs/RoboVoLo）档案

- **GitHub 地址：** <https://github.com/NVlabs/VoLoAgent>（配套基准任务与资产仓：<https://github.com/NVlabs/RoboVoLo>）
- **默认分支：** `main`
- **开源协议：** `NVlabs/VoLoAgent` 为 Apache-2.0；`NVlabs/RoboVoLo` 代码、任务、场景与 NVIDIA 自有资产为 Apache-2.0，第三方贴图为 CC0-1.0，外部 34 类 NVIDIA Isaac SimReady 6.0 资产须另行遵循 NVIDIA Omniverse / Software License Agreement 条款下载
- **关联论文：** *VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation*（[arXiv:2606.07723](https://arxiv.org/abs/2606.07723)，[项目页](https://chicychen.github.io/VoLo/)）
- **开源范围定位：** **已开源** — 编排代理主仓 `NVlabs/VoLoAgent` 与 126 任务基准仓 `NVlabs/RoboVoLo` 均已公开发布
- **核对边界：** 静态代码核对，未本地运行（已核查两个仓库的 `README.md` 与 `files_sample` 目录树、CLI 参数说明、多服务启动命令、协议适配与评测脚本清单；未在本地搭建 Isaac Sim / RoboLab / OpenPI / GraspGen 环境执行）

## 核心入口与目录结构（已核实）

### 1. 主仓库 `NVlabs/VoLoAgent`

- **构建与容器配置：** `pyproject.toml`、`environment.yml`、`Dockerfile`、`Dockerfile.grasp-server-sam3-molmo2`（预置 SAM3、accelerate、tensorflow-cpu、`transformers==4.57.1` 与 `huggingface-hub==0.34.4` 供 Molmo2 侧车使用）。
- **编排器 CLI 与策略代理（`vlm_orchestrator/`）：**
  - `vlm_orchestrator/cli.py`（命令入口 `vlm-orchestrator`）：在默认 `8001` 端口启动代理，连接 `8000` 端口 VLA 策略服务器与 `8003` 端口抓取/感知服务器；支持 `--mode`（`subgoal`、`tool_chain`、`next_goal`、`passthrough` 及归档探索模式 `rewrite`、`adaptive`、`scene_edit`、`subgoal_scene_edit`）、`--failure-monitor`（`vlm`、`gt`、`signal_primary`）、`--recovery-mode`（`replan`、`replan_grasp`、`replan_tools` 等）、`--grasp-seg-mode`（`sam3`、`gdino_sam2`、`molmo_sam2`、`gt_sim`）、`--place-seg-mode`（`molmo_point`、`vlm_point`、`sam3`、`gdino_sam2`、`gt_sim`）、`--motion-planner`（默认 `curobo`，可选 `linear`）、`--enable-stack-mode` 与 `--use-front-camera`。
  - `vlm_orchestrator/failure_handlers/`（`base.py`、`vlm.py`、`gt.py`、`gt_detector.py`、`instruction.py`、`signal.py`、`signal_detector.py`）：VLM 与 GT 故障检测及恢复调度器。
  - `vlm_orchestrator/grasp/`（`server.py`、`client.py`、`tool.py`、`camera.py`、`curobo_planner.py`、`ik.py`、`debug.py`）与 `vlm_orchestrator/motion/`（`planner.py`、`curobo_client.py`）：实现 `grasp()` / `place()` 工具服务端与客户端、深度点云投影、GraspGen 候选生成、cuRobo 避障运动规划与多起点 DLS 逆运动学。
  - `vlm_orchestrator/perception/`（`gdino.py`、`gt_segmentation.py`）、`vlm_orchestrator/gt_metrics/`（`detectors.py`、`events.py`、`manager.py`、`target_resolver.py`）、`vlm_orchestrator/diagnostics/`（`metrics1_failure_logger.py`、`metrics1_gt_rules.py`）与 `vlm_orchestrator/aux_benchmarks/`（`libero_gt.py`、`libero_specs.py`、`robocasa_gt.py`、`robocerebra.py`、`vlabench_gt.py`）。
  - README 另记录了 `vlm_orchestrator/utils/vla_stub_server.py`（`tool_chain` 模式下替代真实 VLA 的桩服务）与 `vlm_orchestrator/utils/molmo2_hf_server.py`（由 `grasp/server.py --enable-molmo` 在 `molmo-env` 中以独立子进程在 `8122` 端口拉起的 Molmo2-8B 侧车服务），以及论文 Appendix C 记录的 `vlm_orchestrator/strategies/`。
- **多基准客户端、本地脚本与测试（`examples/`、`scripts/`、`tests/`、`docs/instructions/`）：**
  - `examples/libero/`、`examples/robocasa/`、`examples/vlabench/`、`examples/droid_sim/`：面向 LIBERO、RoboCasa、VLABench 与 DROID-Sim 的评测客户端和对比脚本（注：使用抓取/放置恢复时非 RoboLab 客户端须传 `--enable-depth`）。
  - `scripts/run_tool_chain_molmo_local.sh`、`scripts/run_stack_ab_local.sh`、`scripts/smoke_gr00t_local.py`、`scripts/smoke_openvla_local.py`、`scripts/smoke_perception_servers.sh`、`tests/test_*.py`。

### 2. 基准仓库 `NVlabs/RoboVoLo`

- **任务与场景包：** `tasks/`、`tasks.json`、`pack.json`、`scenes/`、`images/`、`assets/objects/`（含 `art_shapes`、`chemistry` 与 `math` 三类自定义物体 USD 资产），对应 126 个长时程任务与 123 个场景。
- **安装与资产下载入口：** `download_simready_assets.py --agree-external-terms <RoboLab路径>` 下载 34 类必需 SimReady 6.0 外部资产；`install.py <RoboLab路径>` 将任务、场景和资产安装至 [RoboLab v0.3.0](https://github.com/NVlabs/RoboLab/tree/v0.3.0) 检出目录（需配合 `git lfs pull`）。

## 复现与运行依赖边界

- **多环境并发服务：** 完整运行需分别激活至少 3–4 个隔离 conda 环境：`openpi`（VLA 策略服务，端口 8000，如 $\pi_{0.5}$）、`vlm-orch`（`vlm-orchestrator` 代理，端口 8001，需配置 OpenAI 兼容视觉大模型 `/v1` 接口）、`graspgen`（`vlm_orchestrator/grasp/server.py` 抓取与感知服务，端口 8003，启用 Molmo2 时另需 `molmo-env` 端口 8122），以及 `robolab`（RoboLab v0.3.0 / Isaac Sim 客户端运行 `policies/volo/run.py --remote-port 8001 --enable-subtask --enable-gt-state`）。
- **算力与数据边界：** 按论文 Appendix E，单次仿真试验通常需要约 1 张 L40 GPU（仿真器 + 抓取服务）与 1 张 H100 或第二张 L40（$\pi_{0.5}$；14B DreamZero 需 2 张 H100），全量 126 任务 × 3 试验扫测每套系统约耗 18 GPU-hours；工具链与真机链路高度依赖顶部前视深度图与光追级视觉渲染（如 Isaac Sim PathTracer）。

## 与 wiki 的映射

- 对应实体页：[VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation](../../wiki/entities/paper-agentic-2606-07723.md)
