# Playful-RATs/RATs — Playful Agentic Robot Learning 官方开源仓库档案

- **GitHub 地址：** https://github.com/Playful-RATs/RATs
- **默认分支：** `main`
- **开源协议：** Apache-2.0（RATs 原创代码；第三方依赖 `capx-baseline/` 采用 MIT License，`rats/third_party/` 各子模块保留上游协议）
- **关联论文：** *Playful Agentic Robot Learning*（[arXiv:2606.19419](https://arxiv.org/abs/2606.19419)，CoRL 2026 Spotlight），项目页 [playful-rats.github.io](https://playful-rats.github.io/)
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 预览及 `files_sample` 文件列表核查；需 Python 3.10、`uv`、CUDA GPU、外部模型 API 密钥及 MolmoSpaces / LIBERO-PRO 资产）

## 核心入口与目录结构（已核实）

- **主运行入口（见 `README.md`）：**
  - `scripts/run_rats.py`：自主玩耍（Play）阶段入口，读取环境配置（如 `env_configs/libero/rats_libero_play.yaml`），通过 `--explore --iterations 50 --output-dir outputs/play_libero` 执行玩耍循环并输出技能快照（如 `outputs/play_libero/snapshots/iter050/skills.json`）。
  - `scripts/run_rats_libero_pro_batch.py`：域内批量评测驱动脚本，支持传入 `--seed-skill-library` 加载冻结技能库，通过 `RATS_VERIFIER_STRICT_BENCHMARK=1`、`--gpus`、`--workers` 与 `--skip-completed` 在 LIBERO-PRO 6 个套件上并发运行评测。
- **文档与环境配置（见 `README.md`）：**
  - `docs/setup.md`、`docs/play.md`、`docs/evaluation.md`、`docs/cross-environment.md`：分别覆盖依赖安装、LIBERO-PRO/MolmoSpaces 玩耍配置（含 MolmoSpaces conda 桥接环境）、CaP-X 对比评测以及向 RoboSuite 与真实 Franka Panda 的跨环境迁移流程。
- **第三方运行时子模块（`rats/third_party/`）：**
  - `rats/third_party/LIBERO-PRO`、`rats/third_party/libero_dependencies/robosuite`、`rats/third_party/robosuite`、`rats/third_party/contact_graspnet_pytorch`、`rats/third_party/curobo`、`rats/third_party/sam3`。
- **基线与底层环境/服务栈（`capx-baseline/capx/`，见 `files_sample`）：**
  - **环境与任务适配：** `capx/envs/{launch.py, runner.py, base.py, trial.py}`、`capx/envs/adapters/{libero_wrapper.py, robosuite_wrapper.py}`、`capx/envs/scripts/{run_batch.py, run_libero_batch.py}`、`capx/envs/simulators/{libero.py, franka_real.py, r1pro_b1k.py, robosuite_*.py}` 及 `capx/envs/tasks/franka/*`。
  - **视觉、抓取与运动原语集成：** `capx/integrations/vision/{molmo.py, sam2.py, sam3.py, graspnet.py, owlvit.py}`、`capx/integrations/motion/{curobo.py, curobo_api.py, pyroki.py}`，以及支持技能库注入的接口变体 `capx/integrations/franka/{libero_reduced_skill_library.py, control_reduced_skill_library.py}`。
  - **无状态推理服务端与技能抽取：** `capx/serving/{launch_servers.py, launch_contact_graspnet_server.py, launch_curobo_server.py, launch_pyroki_server.py, launch_sam2_server.py, launch_sam3_server.py, launch_owlvit_server.py, openrouter_server.py, vllm_server.py}`、`capx/llm/{client.py, genai_backend.py}` 及 `capx/skills/{extractor.py, claude_integration.py}`。

## 复现与运行依赖边界

- **系统与环境依赖：** 依赖 Python 3.10（`uv` 管理根 `.venv`，同步 `--extra libero --extra contactgraspnet`）、支持 CUDA 的 GPU、MuJoCo EGL 离屏渲染（`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`CAPX_ENV_STACK=libero`）及预生成 `~/.libero/config.yaml`；MolmoSpaces 需额外 `conda` 桥接环境与数十 GB 场景资产。
- **大模型服务依赖：** 玩耍与推理阶段需配置 `OPENAI_API_KEY`、`GEMINI_API_KEY` 或 `OPENROUTER_API_KEY`（或通过 `capx/serving/` 启动本地 vLLM / OpenRouter 代理及各视觉/IK 微服务）。
- **静态核对边界：** 本档案基于仓库目录树采样、文件清单与官方 `README.md` 完成静态核对，未在本地 GPU 环境下启动仿真或复现实验数值。

## 与 wiki 的映射

- 对应实体页：[Playful Agentic Robot Learning（RATs）](../../wiki/entities/paper-agentic-2606-19419.md)
- 同论文归档：[Playful Agentic Robot Learning 论文核对档案](../papers/paper-agentic-2606-19419.md)
