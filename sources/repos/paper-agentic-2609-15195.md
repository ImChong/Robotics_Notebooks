# AgibotGeneral/harnessvln 官方源码核对归档

- **关联论文：** [HarnessVLN: Unifying Training-Free Embodied Navigation through an Agent Harness](paper-agentic-2609-15195.md)（arXiv:2609.15195v3）
- **仓库地址：** https://github.com/AgibotGeneral/harnessvln
- **默认分支：** `main`
- **许可证说明：** 仓库含 `LICENSE` 与 `THIRD_PARTY.md`；`README.md` 徽章声明为 `CC BY-NC-SA 4.0`（GitHub API `license` 字段返回 `NOASSERTION`），代码基于 LaViRA / Uni-LaViRA 导航基线构建
- **核对边界：** 静态代码核对，未本地运行（基于仓库目录样本 `files_sample` 与 `README.md` 预览核对入口、模块划分与默认配置，未展开逐行 Python 实现或本地执行）

## 已核对的入口与目录结构

- **命令行评估入口（`eval_scripts/`）：**
  - 主调度脚本：`eval_scripts/evaluate.sh`（支持 `R2R`、`R2R_RAND100`、`RXR`、`HM3D_V2`、`HM3D_OVON`、`LANGMAP`）
  - 公共环境变量与参数解析：`eval_scripts/lib/eval_common.sh`、`eval_scripts/lib/runtime_env.sh`、`eval_scripts/egl_env.sh`
  - 单任务快捷脚本：`eval_scripts/vlnce_r2r.sh`、`eval_scripts/vlnce_rxr.sh`、`eval_scripts/hm3d_v2.sh`、`eval_scripts/hm3d_ovon.sh`、`eval_scripts/langmap.sh`
- **Python 多进程与 Harness 运行时（根目录与 `vlnce_baselines/`）：**
  - 多进程启动与评估：`run_mp.py`、`vlnce_baselines/ZS_Evaluator_mp.py`、`vlnce_baselines/agent.py`、`vlnce_baselines/harness_runtime.py`
  - Agent Harness 核心包（`vlnce_baselines/harness/`）：`contracts.py`（接口契约）、`state.py`（共享状态）、`todo.py`（进度/TODO 记忆）、`world_graph.py`（持久化时空图）、`retrieval.py`（任务条件检索）、`geometry.py`（几何可行性与停止校验）、`panorama.py`（四向/全景观测）、`registry.py`（工具注册）、`runtime.py`（运行时闭环）、`trace.py`（执行轨迹追踪）
  - 导航执行、建图与提示词（`vlnce_baselines/`）：`models/Policy.py`、`models/fmm_planner.py`、`models/super_pixel_policy.py`、`models/superpixel_waypoint_selector.py`、`map/mapping.py`、`map/semantic_prediction.py`、`map/RepViTSAM/repvit.py`、`prompts/prompts_vln.py`、`prompts/prompts_objnav.py`、`utils/acyclic_enforcer.py`、`utils/api.py`
  - 跨楼层/楼梯局部策略与深度模块（`navdp/`）：`navdp/policy_agent.py`、`navdp/policy_backbone.py`、`navdp/policy_network.py`、`navdp/depth_anything/depth_anything_v2/`（需单独下载权重 `navdp/navdp-cross-modal.ckpt`）
- **仿真任务扩展与配置（`habitat_extensions/` & `vlnce_baselines/config/`）：**
  - `habitat_extensions/habitat_simulator.py`、`task.py`、`sensors.py`、`measures.py`、`maps.py`、`pose_utils.py`
  - 各任务配置文件：`habitat_extensions/config/{r2r,r2r_rand100,rxr,objectnav_v2,objectnav_ovon,langmap}.yaml` 与 `vlnce_baselines/config/` 同名配置
- **数据构造、评测报告与可视化工具（`tools/` & `server.py`）：**
  - 数据构造与发布校验：`tools/build_r2r_rand100.py`、`tools/build_langmap.py`、`tools/build_rxr_sample.py`、`tools/check_release.py`、`tools/probe_bbox_real.py`
  - 指标汇总：`tools/report_rounds.py`、`tools/report_vln_ne.py`（NE / OSR / NDTW / SR / SPL）、`tools/report_langmap.py`
  - 轨迹可视化服务：`server.py`（启动后在 `http://localhost:9999/?root=eval_data/smoke/rgb` 查看）
- **依赖与兼容补丁（`patches/`）：**
  - `patches/habitat-lab-compat.patch`（针对指定 commit `d6ed1c0a...` 的 `habitat-lab`）
  - `patches/groundingdino-compat.patch`（针对指定 commit `57535c5a...` 的 `GroundingDINO`）

## 复现与运行边界（README 核对结论）

1. **硬件与接口前提：** 需要 Linux + NVIDIA GPU + CUDA 12.8（PyTorch 2.8.0+cu128）+ `habitat-sim=0.3.1`，并在 `.env` 中配置外部多模态聊天补全接口（`MODEL_API_KEY`、`MODEL_BASE_URL`、`MODEL_NAME`）。
2. **外部资产需自行申请：** Matterport3D（需签署协议申请）、HM3D v0.2（val 切分）、GroundingDINO/SAM/NavDP 权重均需按 `THIRD_PARTY.md` 从上游下载；仓库内仅自带分层抽样清单（如 `data/datasets/stratified_samples/smoke_3.json`）。
3. **默认开关差异：** `README.md` 明确指出，证据化停止校验（`EVAL.STOP_GEOMETRIC_GATE` 与 `EVAL.STOP_AUTOSTOP_BEFORE_TIMEOUT`）在默认运行命令中处于**关闭**状态，须显式传参开启才会启用几何门控。
4. **真机部署代码边界：** `files_sample` 与 `README.md` 主要覆盖 Habitat 仿真评测（含论文未单列结果的 `LANGMAP` 基准）；AGIBOT A3 Ultra 人形机器人部署栈（Fast-FoundationStereo、NVIDIA Thor 局部控制桥接）未在本次目录样本中见独立真机驱动目录。
