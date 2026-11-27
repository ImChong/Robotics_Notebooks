# lty759/SafeMem 官方代码核查

- **代码仓库：** <https://github.com/lty759/SafeMem>
- **开源协议与分支：** MIT License，默认分支 `main`（仓库 README 标注 2026-09-11 发布代码）
- **对应论文：** [Safe Task Planning with Long-Term Graph Memory for Embodied Agents (arXiv:2609.08444v1)](../papers/paper-agentic-2609-08444.md)
- **配套依赖仓：** [AI45Lab/IS-Bench](https://github.com/AI45Lab/IS-Bench)（提供基础仿真环境、任务与 BDDL/场景元数据）、[StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)（OmniGibson 底层物理仿真）、[facebookresearch/sam2](https://github.com/facebookresearch/sam2)（实例掩码分割，权重 `sam2.1_hiera_large.pt`）
- **核对范围：** 静态核对 GitHub 仓库元数据、`README.md` 全文预览与完整 `files_sample` 目录树清单（含 `og_ego_prim/`、`scene_graph/`、`configs/prompts/risk.txt`、`entrypoints/`、`generation/`、`run_benchmark.sh`）。静态代码核对，未本地运行。
- **通读：** 未完成
- **开源范围判断：** **已开源** OmniGibson / IS-Bench 仿真评测全链路源码，涵盖命令行评测入口、部分可观测遮挡设置、基于 `gemini`/`gpt`/`qwen` 与 SAM2 的增量场景图工作进程、风险预测器、重规划循环及提示词模板；Franka Panda 真机底层运动原语与 D405 相机控制端代码未见于当前公开目录清单。另论文正文发现列表中的 `google/safevalues` 为无关前端安全库，非本项目代码。

## 已核对目录与入口

1. **命令行运行入口与封装脚本：**
   - `og_ego_prim/cli/online_benchmark_once.py`：单任务评测主入口，支持 `--use_scene_graph`、`--occlude_risk_source`、`--use_risk_predictor`、`--risk_check_probability`（对应论文保守因子 $\eta$，示例为 `0.8`）、`--risk_max_replans`（示例 `10`）、`--model`、`--risk_predictor_model` 等参数；`run_benchmark.sh` 对此命令做了便捷封装。
   - `og_ego_prim/cli/online_benchmark_all.py`：按风险类别任务清单（`entrypoints/electric.txt`、`entrypoints/fire.txt`、`entrypoints/falling.txt`、`entrypoints/food.txt`）批量运行评测，输出每步观测、场景图快照、单任务 `report.json` 及汇总文件 `report_<model>_all.json`。
   - `og_ego_prim/cli/summary_benchmark_results.py`：汇总评测结果统计。
2. **规划、风险预测与评测核心 `og_ego_prim/`：**
   - 基准与状态跟踪：`og_ego_prim/benchmark/base_benchmark.py`、`online_benchmark.py`、`offline_benchmark.py`、`fixed_risk_occlusion.py`、`custom_behavior_task.py`、`evaluator/evaluator.py`、`evaluator/bddl_goal_condition.py`、`tracker/online_tracker.py`。
   - 规划器与模型后端：`og_ego_prim/models/plan_agent.py`、`base_client.py`、`server_inference.py`、`hf_inference.py`、`image_utils.py`。
   - 风险预测与场景图客户端：`og_ego_prim/risk_predictor.py`（读取 `configs/prompts/risk.txt`）、`og_ego_prim/scene_graph_client.py`。
   - 原子技能原语执行：`og_ego_prim/primitives/ego_primitives.py`、`executor.py`、`attachment.py`、`object_states_utils.py`、`primitive_utils.py`。
3. **增量场景图模块 `scene_graph/`：**
   - 进程通信与会话：`scene_graph/online_worker.py`（通过 stdin/stdout 供 `scene_graph_client.py` 调用）、`scene_graph/vidsgg.py`（会话初始化与更新）、`scene_graph/run_vidsgg.py`。
   - 分割匹配与图合并：`scene_graph/mask.py`（SAM2 掩码生成与跨帧匹配）、`scene_graph/online_utils.py`（对象节点与关系合并）、`scene_graph/Object.py`、`scene_graph/vis.py`。
   - VLM 后端与提示词：`scene_graph/vlms/`（`gemini.py`、`gpt.py`、`qwen.py`）与 `scene_graph/vlms/prompts/`（`generate_frame_scene_graph.txt`、`generate_rels.txt`、`classify_and_describe_mask.txt`）。
4. **安全标注生成工具 `generation/`：**
   - `0_extract_safety_principle.py`、`1_detect_risk.py`、`2_check_risk.py`、`3_generate_safety_condition.py`、`4_generate_golden_planning.py`、`extract_wash_rules.py`。

## 边界说明

- 本归档仅基于仓库公开快照的目录文件列表与 `README.md` 说明进行静态结构核对，**未在本地配置 IS-Bench / OmniGibson 仿真环境、SAM2 权重（`sam2.1_hiera_large.pt`）或 OpenAI / Gemini API 进行实际运行**。
- `README.md` 指出任务定义（`data/tasks`、`data/bddl`）、场景元数据（`data/metas`、`data/cameras`）及 OmniGibson 资产不在本仓库内，需先按 IS-Bench 文档配置后放入 `data/` 目录。

## 映射

- [SafeMem 论文实体与源码运行时序图](../../wiki/entities/paper-agentic-2609-08444.md)
