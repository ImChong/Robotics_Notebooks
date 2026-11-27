# nepfaff/scenesmith — SceneSmith 官方开源仓库档案

- **GitHub 地址：** https://github.com/nepfaff/scenesmith
- **默认分支：** `main`
- **开源协议：** MIT License
- **关联论文：** *SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes*（[arXiv:2602.09153](https://arxiv.org/abs/2602.09153)，ICML 2026 Spotlight）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 核查，未在本地实际安装 SAM3D 权重或执行场景生成）。

## 核心入口与目录结构（已核实）

- **主运行入口与配置：**
  - `main.py`：Hydra 驱动的场景生成主入口（README 运行示例：`python main.py +name=my_experiment` 或容器内 `docker compose run --rm scenesmith python main.py +name=my_experiment`）。
  - `configurations/config.yaml`、`configurations/experiment/indoor_scene_generation.yaml`
  - 五阶段智能体配置：`configurations/floor_plan_agent/stateful_floor_plan_agent.yaml`、`configurations/furniture_agent/stateful_furniture_agent.yaml`、`configurations/wall_agent/stateful_wall_agent.yaml`、`configurations/ceiling_agent/stateful_ceiling_agent.yaml`、`configurations/manipuland_agent/stateful_manipuland_agent.yaml`（对应各自的 `base_*` 配置）。
- **核心智能体与场景状态：**
  - `scenesmith/agent_utils/base_stateful_agent.py`：有状态 VLM 智能体基类与会话记忆管理。
  - `scenesmith/agent_utils/house.py`、`scenesmith/agent_utils/checkpoint_state.py`、`scenesmith/agent_utils/action_logger.py`：层次化房屋/房间状态表示与检查点回滚机制。
  - `scenesmith/agent_utils/drake_utils.py`、`scenesmith/agent_utils/clearance_zones.py`、`scenesmith/agent_utils/articulated_physics_analyzer.py`：Drake 物理仿真、碰撞/净空检查与关节物理分析工具。
- **资产管理与多服务端模块：**
  - `scenesmith/agent_utils/asset_manager.py`、`scenesmith/agent_utils/asset_registry.py`、`scenesmith/agent_utils/asset_router/router.py`：VLM 资产路由器（阶段过滤、复合拆解与策略分发）。
  - `scenesmith/agent_utils/geometry_generation_server/`：`server_manager.py`、`server_app.py`、`worker_pool.py`、`gpu_worker.py`、`sam3d_pipeline_manager.py`、`hunyuan3d_pipeline_manager.py`（支持 SAM3D 默认后端与 Hunyuan3D-2 备选后端的多 GPU 异步生成服务）。
  - `scenesmith/agent_utils/articulated_retrieval_server/` 与 `scenesmith/agent_utils/hssd_retrieval_server/`：ArtVIP / PartNet-Mobility / HSSD 检索服务端与 CLIP 相似度匹配（`retrieval.py`）。
  - `scenesmith/agent_utils/convex_decomposition_server/`：V-HACD 凸分解服务端。
  - `scenesmith/agent_utils/blender/`：`server_manager.py`、`renderer.py`、`surface_rendering_mixin.py`、`view_generation_mixin.py`、`wall_utils.py`、`surface_utils.py`（支持 `bubblewrap` 单 GPU 隔离的 EEVEE Next 渲染服务端）。
- **容器与依赖：**
  - `pyproject.toml`（使用 `uv sync` 管理）、`Dockerfile`、`docker-compose.yaml`。

## 复现与运行依赖边界

- **API 密钥：** 需设置 `OPENAI_API_KEY`（GPT-5 系列智能体及默认图像生成），可选 `GOOGLE_API_KEY`（Gemini 图像生成后端）。
- **模型权重与外部数据：** 默认配置依赖 HuggingFace 授权下载 `facebook/sam3` 与 `facebook/sam-3d-objects`（约 5 GB，建议至少 32 GB 显存 GPU；若改用 Hunyuan3D-2 可压缩至 24 GB 显存，或改用 `strategy: "hssd"` 纯检索免除生成后端）、预处理 ArtVIP 数据集（`huggingface-cli download nepfaff/scenesmith-preprocessed-data`）以及 ambientCG PBR 材质；可选接入 HSSD（~72 GB）、Objaverse ObjectThor（~50 GB）或 PartNet-Mobility。
- **物理与渲染依赖：** 依赖 Drake 进行场景物理验证、非穿透投影与重力沉降；多 GPU 并行渲染可选安装 `bubblewrap` 避免多进程抢占 GPU 0 导致显存溢出。

## 与 wiki 的映射

- 对应实体页：[SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes](../../wiki/entities/paper-agentic-2602-09153.md)
