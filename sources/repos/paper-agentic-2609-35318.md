# dexagent1/DexAgent 官方代码核查

- **代码仓库：** <https://github.com/dexagent1/DexAgent>
- **开源协议与分支：** MIT 协议，默认分支 `main`（仓库快照时 60 Stars）
- **对应论文：** [DexAgent: An Agentic Human2Sim2Robot Framework for Dexterous Manipulation with Self-Evolving Tool Library (arXiv:2609.35318v2)](../papers/paper-agentic-2609-35318.md)
- **核心依赖与可选子环境：** 主环境（Python 3.11 + `requirements.txt`）固定 **MuJoCo 3.10** 与 **NumPy 1.x** 以保证接触仿真一致性；感知技能（SAM 3、SAM 2、HaMeR、Video Depth Anything、GeoCalib、SAM 3D）、可变形绳索仿真（**Isaac Sim 5.0**）、GPU 抓取先验（**BODex**、**DexGraspNet**）及 Stage 4 数据重绘（**VOID**、**Blender `bpy`**，需约 40 GB GPU 显存）均通过 `dexagent.yaml` 的 `environments:` 与 `models:` 挂载独立环境。
- **核对范围：** 静态核对 GitHub 仓库元数据、`README.md` 全文预览与 `files_sample` 目录文件清单（覆盖 `dexagent.yaml`、`agent/` 运行时模块与 30 个测试文件及 `examples/_assets/` 示例网格资产，并结合 `README.md` 核对 `run_dexagent.py`、四个阶段目录脚本与 `tool_library/`/`setup/`/`library_visualizer/` 结构）。静态代码核对，未本地运行。
- **通读：** 未完成
- **开源范围判断：** **已开源** 面向 VLM 工具调用与自进化工具库的智能体运行时框架（`agent/`）、四阶段执行脚本与配置模板（`dexagent.yaml`、`setup/`）以及工具库可视化模块；运行完整四阶段需自备 VLM API 密钥或订阅子智能体会话、Hugging Face 权重缓存与上述 GPU 子环境。

## 已核对目录与入口

1. **统一编排与四阶段 CLI 入口（依据 `README.md`）：**
   - `run_dexagent.py dexagent.yaml`：按 Stage 1 → Stage 4 顺序串联执行完整流程，支持 `--from stage3`、`--to stage2` 等断点续跑与区间截断。
   - `stage1_semantic_understanding/run_stage1.py dexagent.yaml`：输入示范视频与提示词，产出 `scene_semantics.json`。
   - `stage2_scene_reconstruction/run_stage2.py dexagent.yaml`：构建并通过属性校验的仿真场景 `scene.xml`（绳索等可变形体调用 Isaac Sim 5.0）。
   - `stage3_trajectory_generation/run_stage3.py dexagent.yaml`：依据子目标生成经过物理校验的机器人单回合轨迹并打包输出 `runs/<name>/stage3/stage4_task/stage4.json`。
   - `stage4_robot_data_generation/run_stage4.py <stage4.json>`：执行状态随机化与 VOID + Blender 重贴图渲染，输出至 `stage4_task/outputs/`。
2. **智能体运行时与工具库治理内核（依据 `files_sample`）：**
   - 核心调度与模型调用：`agent/runner.py`、`agent/llm.py`、`agent/toolcall.py`、`agent/devcall.py`、`agent/devtools.py`、`agent/config.py`、`agent/budget.py`（控制单阶段 `max_rounds` 与 `agent.budget_usd` 预算上限）、`agent/console.py`。
   - 自进化工具库与复用修复：`agent/library.py`、`agent/reuse.py`、`agent/repair.py`、`agent/verification_feedback.py`、`agent/search_progress.py`、`agent/integrity.py`、`agent/artifacts.py`、`agent/reference_evidence.py`、`agent/input_identity.py`、`agent/external.py`。
   - 提示词与说明：`agent/prompts/system.md`、`agent/prompts/building_tools.md`、`agent/render_prompt.py`、`agent/README.md`。
   - 单元测试套件（`agent/tests/`，共 30 个测试文件）：覆盖预算门控（`test_budget.py`、`test_result_hooks_and_wall_budget.py`）、工具发现与检索排序（`test_tool_discovery.py`、`test_library_search_ranking.py`、`test_library_schema_types.py`）、跨阶段修复与 Stage 3 恢复（`test_cross_stage_repair.py`、`test_stage3_recovery.py`、`test_pipeline_stage3_resume.py`）、校验反馈与阻塞停止（`test_verification_feedback.py`、`test_blocked_stop.py`、`test_report_blocked.py`）以及示例/轨迹复用（`test_example_trajectory_reuse.py`）。
3. **工具库、默认台架与示例资产：**
   - `tool_library/`：存放自进化沉淀的 Skill 与 Verifier 实现（在线展厅位于 <https://dexagent1.github.io/library/>，本地对应 `library_visualizer/`）。
   - `setup/`：默认标定的实验台架配置、第一人称相机参数与机器人模型；`examples/_assets/` 包含预置 `.obj`/`.stl` 网格资产。

## 边界说明

- 本归档仅基于仓库公开元数据、`README.md` 全文预览与 `files_sample` 文件清单完成静态结构核对，**未在本地构建 Conda 环境、连接 VLM API 或启动 MuJoCo / Isaac Sim / Blender 进行实跑验证**。
- `README.md` 明确提示各阶段涉及多轮 VLM 工具调用循环，运行耗时与 API 费用受视频长度、模型选择及工具库命中率影响较大，首次运行前需在 `dexagent.yaml` 中设定 `max_rounds` 与 `agent.budget_usd`。

## 映射

- [DexAgent 论文实体与源码运行时序图](../../wiki/entities/paper-agentic-2609-35318.md)
