# Yomier/AGOS-Bench — AGOS-Bench / AGOS-Dataset / AGOS-Agent 官方仓库档案

- **GitHub 地址：** https://github.com/Yomier/AGOS-Bench
- **默认分支：** `main`
- **开源协议：** 代码仓库采用 Apache-2.0；配套数据集 [`Yomier/AGOS-Dataset`](https://huggingface.co/datasets/Yomier/AGOS-Dataset)（Hugging Face / ModelScope）采用 CC BY 4.0
- **关联论文：** *Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method*（[arXiv:2609.08402](https://arxiv.org/abs/2609.08402)）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 目录清单核查；未在本地安装 CARLA-Air v0.1.7 仿真器或启动 GPU 推理与感知服务）。

## 核心入口与目录结构（已核实）

- **基准评测与指标入口（`benchmark/`，`README.md` 已核实命令行）：**
  - `python -m benchmark.run_parallel_eval`：支持 `--dry-run` 离线校验任务计划、参考图像与导航资产，以及指定 `--arm`（`agos_b`、`baseline_a`、`menu_oracle`、`minimal_agent`）与 `--split test300` 的实时并行评测。
  - `python -m benchmark.paper_metrics`：读取锁定结果，排除基础设施故障后计算论文全部 9 项指标与对应分母。
- **Agent 配置与基线矩阵（`agents/configs/`）：**
  - `agos_a.yaml`、`agos_b.yaml`、`agos_c.yaml`（对应论文主方案 AGOS-Agent 变体，README 评测入口使用 `agos_b`）、`baseline_a.yaml`（Baseline-Agent）、`menu_oracle.yaml`（零模型调用的部署冒烟测试配置）、`minimal_agent.yaml`（Minimal-Agent 直接多模态控制）。
- **核心算法与协同策略包（`agos/`）：**
  - `agos/runtime/`：`runner.py`（任务闭环主循环）、`evaluation_observations.py`、`observation_maps.py`、`paper_audit.py`。
  - `agos/methods/cooperative_search/`：`method.py`、`controller.py`、`coverage.py`（联合覆盖分配）、`policy.py`（事件驱动决策与 MLLM 调用）、`perception.py`、`contracts.py`。
  - `agos/methods/common/`：`grounded_localizer.py`、`online_map.py`、`route_tool.py`、`uav_annotation.py`、`ugv_sixview.py`、`source_guard.py`、`legacy_sixview_collage.py`。
  - `agos/methods/menu_oracle/method.py`、`agos/methods/minimal_agent/method.py`、`agos/methods/vln_os_common/contracts.py`、`agos/methods/registry.py`。
  - `agos/perception/`：`perception.py`（Grounding DINO + SigLIP 2 服务调用）、`uav_grid_geometry.py`（像素反投影到世界坐标）、`public_road_elevation.py`。
  - `agos/roadnet/`：`router.py`、`locator.py`、`candidate_approach.py`、`option_provider.py`、`anchors.py`、`store.py`、`adapter.py`、`registry.py`、`schema.py`、`serializer.py`。
  - `agos/control/carla_interface.py`、`agos/agents/`（`agents.py`、`http_accounting.py`）、`agos/eval/`（`eval.py`、`metric_contract.py`、`metric_primitives.py`、`navigation_oracle.py`、`deployment_validation.py`、`contamination.py`、`source_compatibility.py`）。
  - `agos/scripts/semantic_nav_adapter.py`、`agos/episode/`（`episode_io.py`、`reference_photo.py`、`vehicle_grounding.py`）、`agos/maps/waypoints.py`、`agos/benchmark_protocol.py`、`agos/benchmark_region.py`。
- **路网与标定资产（`assets/`）：**
  - `assets/calibration/uav_topdown_80m_yaw-90_fov90.json`：80 m 俯视相机标定参数。
  - `assets/navigation/road_graphs_v2/Town01..Town05, Town10HD/`：6 张地图冻结的 `decision_nodes.jsonl.gz`、`decision_edges.jsonl.gz`、`fine_nodes.jsonl.gz`、`fine_edges.jsonl.gz`、`manifest.json` 与 `FROZEN` 标记。
- **数据集与微调训练入口（`README.md` 已核实模块）：**
  - `dataset/download.py` 与 `dataset/verify.py`：从 Hugging Face 或 ModelScope 下载并做 SHA-256 校验（含 `train1100`、`validation275`、`seentest450`、`taskset-test300`、`certification`、`trajectories-train1100` 六个分卷）。
  - `training/corpus.py` 与 `training/train.py`：将公开的 `trajectories-train1100` 构造为训练语料并按 rank-8、alpha-16、5,120-token 过滤执行 Qwen3.5-4B 联合 LoRA 微调。
  - `carlaair/serving/vllm`：挂载训练后 LoRA adapter 的 vLLM 推理服务入口。

## 复现与运行依赖边界

- **多环境拆分：** Ubuntu 22.04 x86_64 + NVIDIA GPU；实时仿真需安装上游 CARLA-Air v0.1.7（`carlaAir` Python 3.10 环境）；感知服务需 Python 3.12（`agos-perception`，加载 Grounding DINO、SigLIP 2 与可选 SAM2.1 快照）；推理服务需 Python 3.12（`agos-vllm`，`vllm==0.28.0`、`transformers==5.16.1`，加载 `Qwen3.5-4B` 或兼容 OpenAI 图像 API）。
- **权重与语料边界：** README 明确说明论文原版实验使用的 LoRA 权重及当时的 301 回合（6,014 条）历史训练输入不在发布范围内；复现需基于公开发布的 `trajectories-train1100` 经 `training.corpus` 与 `training.train` 重新训练获得 adapter。
- **离线冒烟验证：** 无 GPU 或未装仿真器时，可用 Python 3.10 执行 `python -m benchmark.run_parallel_eval ... --arm menu_oracle --dry-run` 校验本地数据卷与路网资产完整性。

## 与 wiki 的映射

- 对应实体页：[Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method](../../wiki/entities/paper-agentic-2609-08402.md)
