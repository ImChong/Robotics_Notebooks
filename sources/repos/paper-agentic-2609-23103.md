# diaggen/diaggen 官方代码核查

- **代码仓库：** <https://github.com/diaggen/diaggen>
- **开源协议与分支：** 默认分支 `master`（GitHub API 元数据未返回开源协议字段 `license`）
- **对应论文：** [DiagGen: Agentic Generation of Deformable Assets with Sim-based Diagnostics for Robotic Simulation (arXiv:2609.23103v1)](../papers/paper-agentic-2609-23103.md)
- **核心依赖与第三方子树：** 仓库 `third_party/` 目录及 `README.md` 声明绑定定制版 **Genesis 1.2.0**（提供实时诊断 API）、**SAM3** 与 **OmniPart**（`third_party/OmniPart/` 已随仓包含部分源码），运行时按 `hag4r`、`hag4r_mesh`、`hag4r_segmentation`、`omnipart` 四个 Conda 环境划分。
- **核对范围：** 静态核对 GitHub 仓库元数据、`README.md` 全文预览与 `files_sample` 目录文件清单（覆盖 `.agents/skills/run-diaggen-pipeline/`、`.codex/config.toml`、`configs/`、`hag4r/agentic/`、`hag4r/tools/`、`scripts/` 与 `third_party/OmniPart/`）。静态代码核对，未本地运行。
- **通读：** 未完成
- **开源范围判断：** **已开源** 面向 Agent 技能编排（`.agents/skills/`）与 Python 确定性工具/MCP 服务（`hag4r/`）的完整资产生成与仿真诊断主干源码；模型检查点（checkpoints）与本地 Conda 依赖需按 `requirements.txt` 与 `configs/` 自行准备；`github_urls` 中同时出现的 `nerfies/nerfies.github.io` 为项目主页网页模板仓库，非本项目代码。

## 已核对目录与入口

1. **流水线主入口与配置：**
   - `hag4r/agentic/run_diaggen_pipeline.py`：单资产初始化与状态账本（ledger）构建入口，通过 `--source_image`、`--run_id`、`--run_root`、`--config`（如 `configs/config_genesis_diagnostics_enabled.yaml` 与对比基线 `configs/config_genesis_diagnostics_disabled.yaml`）启动流程。
   - `.agents/skills/run-diaggen-pipeline/SKILL.md`：总控 Agent Skill 入口，向下游阶段分派任务；保留 `hag4r` 命名空间以兼容底层依赖接口。
2. **Agent 技能套件（`.agents/skills/run-diaggen-pipeline/`）：**
   - 前置生成阶段：`image-cleanup/`（含 `cleanup-validation/`）、`segmentation/`、`omnipart/`、`material-inference/`（拆分为 `geometry-analysis`、`material-semantics`、`fill-mode-decision`、`physical-parameter-estimation`、`material-validation-repair`）、`mesh-processing/`。
   - 仿真诊断阶段：`genesis-diagnostics/`（包含 `session-planning/`、`probe-targeting/`、`episode-authoring/`、`live-probing/`、`session-synthesis/` 及 `scripts/update_diagnostic_faults.py`）。
   - 后处理与导出：`post-mesh-texture/`、`final-export/`。
3. **Agent 运行时、MCP 服务与确定性工具（`hag4r/`）：**
   - 诊断与控制：`hag4r/agentic/diagnostic_mcp_server.py`（由 `.codex/config.toml` 与环境变量 `DIAGGEN_ROOT` 挂载）、`diagnostic_session_registry.py`、`diagnostic_force_limited_controller.py`、`diagnostic_cues.py`、`genesis_vlm_schemas.py`。
   - 运行时工具链：`hag4r/agentic/runtime_tools/diagnostics.py`、`pipeline_transition.py`、`stage_tools.py`、`stage_terminal.py`，配合 `runtime_state.py`、`state.py`、`skill_loader.py`、`prompting.py`、`run_config.py`。
   - 领域工具：`hag4r/tools/` 下的 `image_cleanup.py`、`segmentation.py`、`omnipart.py`、`material_inference.py`、`mesh_processing.py`、`metric_mesh_scaling.py`、`volumetric_meshing.py`、`probe_targeting.py`、`part_grounding.py`、`triple_view_evidence.py`、`post_mesh_texture.py`、`final_export.py`，以及 `hag4r/tools/genesis/`（`live_client.py`、`live_protocol.py`、`diagnostic_timing.py`、`config.py`）。
   - 辅助脚本：`scripts/assign_params_to_prims.py`。

## 边界说明

- 本归档仅基于公开仓库目录清单与 `README.md` 预览完成静态结构核对，**未在本地配置 `.conda/hag4r*` 环境、模型检查点、VLM API 或 Genesis 实时仿真服务进行运行验证**。
- `README.md` 指出运行时依赖的 Genesis、SAM3、OmniPart 模型权重路径需本地配置，且目前入口针对单资产账本逐一运行，尚未验证多资产高并发批处理。

## 映射

- [DiagGen 论文实体与源码运行时序图](../../wiki/entities/paper-agentic-2609-23103.md)
