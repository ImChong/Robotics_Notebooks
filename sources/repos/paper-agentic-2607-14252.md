# yuzihaowashu/MEMORA — MEMORA 官方开源仓库档案

- **GitHub 地址：** https://github.com/yuzihaowashu/MEMORA
- **默认分支：** `main`
- **开源协议：** MIT License（源码）；派生基准任务与 HuggingFace 可下载参与者记忆/评估输出遵循 EPIC-KITCHENS CC BY-NC 4.0 协议
- **关联论文：** *MEMORA: Embodied Action Memory from Egocentric Videos for Reasoning and Planning*（[arXiv:2607.14252](https://arxiv.org/abs/2607.14252)，RSS 2026 FM4RoboPlan Oral / EMNLP 2026）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 核查，未在本地下载 HuggingFace 数据包或部署 vLLM/GPU 推理服务）。

## 核心入口与目录结构（已核实）

- **打包配置与文档入口：**
  - `pyproject.toml`：定义项目依赖及公开 CLI 入口（`memora-segment-encode`、`memora-memory-edit`、`memora-plan`、`memora-eam-qa`）。
  - `docs/README.md`、`docs/SETUP.md`、`docs/DATA.md`、`docs/PIPELINE.md`、`docs/REPRODUCE.md`：环境安装、数据校验、端到端流水线契约与论文指标复现指南。
  - `scripts/setup_environment.sh`（支持 `analysis` CPU 复现环境与 `gpu` 推理环境）、`scripts/configure_paths.sh`。
- **记忆构建管线（`src/memora/pipeline/`）：**
  - `formation_config.py`、`api_client.py`：分段编码与模型服务接口配置。
  - `memory_editor/cli.py`、`memory_editor/flat_1d.py`、`memory_editor/model_response.py`：在线记忆编辑器 CLI（`memora-memory-edit`），输出 `participant_memory.json` 及 Flat-1D 对照存储。
  - `consolidation/runner.py`、`consolidation/evidence.py`、`consolidation/prompts.py`、`consolidation/schema.py`：离线巩固与 Inferred Knowledge 丰富模块（经 `--run-offline-consolidation` 触发）。
- **记忆智能体与多存储检索工具（`src/memora/memory_agent/`）：**
  - `agent.py`、`agent_environment.py`、`tool_call_parsing.py`：ReAct 代理循环及本地工具调用解析。
  - `memory_representations/{memora,flat,graph}/{prompts.py,tools.py}` 与 `shared.py`：MEMORA、Flat-1D、Graph-2D 三类表示的读接口与 Prompt 隔离封装。
  - `tools/stores/{environment.py,entity.py,entity_normalization.py,activity.py,inferred.py}`：四类型存储底层实现。
  - `tools/{tool_interface.py,context.py,embedding.py,ranking.py,formatting.py,lexicon.py,schemas.py}` 及 `tools/evidence/{cross_store.py,episode.py,planning.py,semantic.py}`：基于 `e5-base-v2` 的向量检索、余弦 $\ge 0.95$ 去重与跨存储证据工具。
- **评测与基准（`src/memora/evaluation/` & `src/memora_bench/`）：**
  - `evaluation/eam_qa/{runner.py,tasks.py,prompts.py,answers.py}`：EAM-QA 评测入口（`memora-eam-qa run`）。
  - `evaluation/planning/{runner.py,planning_environment.py,tasks.py,prompts.py,context.py,parser.py,step_checks.py}` 与 `prompt_profiles/{memora.py,flat_1d.py,graph_2d.py,no_memory.py,shared.py}`：MEMORA-Planning 评测与计划解析器（`memora-plan`）。
  - `evaluation/results.py`、`evaluation/settings.py`、`scripts/build_evaluation_runs.py`。
- **论文结果复现与基准构建脚本（`scripts/`）：**
  - `scripts/paper_results/{reproduce_results.py,eam_qa_metrics.py,planning_metrics.py}` 及 `scripts/paper_results/commands/{eam_qa,planning_replay,planning_generalize}/*.sh`：直接基于已保存推理输出复算论文主表指标。
  - `scripts/benchmark_construction/{extract_replay_candidates.py,build_replay_suite.py,generate_generalize_candidates.py,build_generalize_suite.py}`：Replay 与 Generalize 基准构造脚本。

## 复现与运行依赖边界

- **Python 版本与双模式环境：** 需 Python 3.10–3.12（`setup_environment.sh` 创建 `.venv` 并以可编辑模式安装包）。仅复算论文表格使用 `analysis` 模式（纯 CPU）；从视频构建记忆或运行本地模型推理需 `gpu` 模式并配合 CUDA GPU 与 vLLM。
- **数据与模型权重：** 18 名参与者四存储记忆及已保存模型输出可通过 `scripts/data/download.py` 从 HuggingFace `DependableDavid/MEMORA` 获取；若要从头运行阶段 1 编码（`memora-segment-encode`），需自行按官方许可获取 EPIC-KITCHENS-100 原始第一人称 RGB 视频，并准备 `Qwen2.5-Omni-7B`、`Qwen3-30B-A3B-Instruct-2507`、`Qwen3.6-35B-A3B`、`intfloat/e5-base-v2` 及回答模型（或兼容 API 凭证）。

## 与 wiki 的映射

- 对应实体页：[MEMORA: Embodied Action Memory from Egocentric Videos for Reasoning and Planning](../../wiki/entities/paper-agentic-2607-14252.md)
