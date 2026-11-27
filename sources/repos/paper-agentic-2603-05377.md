# OpenFrontier 官方代码仓库

> 来源归档（repo）

- **标题：** OpenFrontier: General Navigation with Visual-Language Grounded Frontiers
- **类型：** repo（零样本视觉前沿语义锚定与导航评测/演示框架）
- **链接：** <https://github.com/cvg/OpenFrontier>
- **论文：** <https://arxiv.org/abs/2603.05377>（见 [论文来源归档](../papers/paper-agentic-2603-05377.md)）
- **项目页：** <https://boysun045.github.io/OpenFrontier-Project/>
- **默认分支：** `main`
- **许可证：** 仓库根目录未见独立 `LICENSE` 文件（GitHub API 元数据未登记许可证，使用前需以仓库后续更新或作者声明为准）
- **核对边界：** 静态代码核对，未本地运行（基于仓库 README 与目录/文件采样核对）
- **沉淀到 wiki：** [wiki/entities/paper-agentic-2603-05377.md](../../wiki/entities/paper-agentic-2603-05377.md)

## 仓库结构与运行入口（依据 README 与文件列表采样）

| 路径 / 入口 | 用途 |
|---|---|
| `benchmark.py` | Habitat 基准评测主入口，支持 `--benchmark hm3d \| mp3d \| ovon`、`--eval_episodes`、`--split` 分片并行 |
| `demo_navigation.py` / `demo_exploration.py` | Open3D 轻量交互式导航与纯探索演示脚本（注：README 命令示例写作 `demo.py`，实际仓库文件为 `demo_navigation.py` 与 `demo_exploration.py`） |
| `sam3_server.py` / `scripts/run_sam3.sh` | 基于 Flask 与 `third_party/sam3` 子模块的 SAM3 开放词汇分割服务端 |
| `vlm_server.py` / `vlm/` | 本地 VLM 推理服务端（监听 `12185` 端口，支持 `InternVL3_5-8B`、`LLaVA`、Gemma）及 `vlm/client.py` 统一客户端（含 Gemini/Gemma API） |
| `config/` | 系统参数配置，含 `config/navigation.yaml`、`config/exploration.yaml`、`config/any.yaml`、`config/scannetpp.yaml` |
| `frontier/` | 视觉前沿检测与管理核心：`detector.py`、`frontier.py`、`graph.py`、`manager.py` 及 `frontier/model/predict.py`（UNet/ResNet FrontierNet） |
| `nav/` | 智能体封装：`agent.py`、`habitat_agent.py`、`o3d_agent.py`、`pointnav_agent.py`、`detected_object.py` |
| `planner/` | 局部/全局路径规划器：`pointnav_planner.py`、`habitat_planner.py`、`habitat_server_planner.py`、`occ_rrt_point3d.py`、`astar.py`、`occupancy_grid.py` |
| `mapping/wavemap.py`、`mono_depth/` | 真机/体素建图与单目深度估计接口（`Metric3D.py`、`UniK3D.py`） |
| `vlfm/`、`metrics.py`、`scripts/download_weights.sh` | 复用 PointNav 策略工具（`vlfm/policy/utils/pointnav_policy.py`）、评测指标及权重下载脚本 |

## 复现条件与开源边界

- **双环境路线：**
  - **Open3D Demo 环境**：Python 3.11 + `requirements.txt`，无需 Habitat，可快速验证前沿检测与单场景导航（示例网格 `examples/mv2HUxq3B53.glb`）。
  - **Habitat 评测环境**：需独立安装 `facebookresearch/habitat-lab`（`v0.2.4` 分支，Python 3.9）与 `habitat-sim`，再安装 `requirements_habitat.txt`，并配置 `OPENFRONTIER_DATA_ROOT`（HM3D v0.2 / MP3D / OVON）与 `HABITAT_LAB_ROOT`。
- **外部模型与鉴权依赖：**
  - `sam3_server.py` 需在独立 Python 3.12 环境安装 `third_party/sam3` 子模块，并在 Hugging Face 申请 `facebook/sam3` 权重访问权限后执行 `hf auth login`。
  - VLM 支持配置 `GOOGLE_API_KEY` / `GEMINI_API_KEY` 调用 Gemini-2.5-flash，或启动 `vlm_server.py` 加载本地开源 VLM。
  - 真机 ROS 部署代码与机器人端控制接口未见于公开目录采样中（论文 §S1-D 描述基于 ROS + Boston Dynamics Spot + Wavemap + OMPL $RRT^\star$，公开仓库提供 `mapping/wavemap.py` 与 `planner/occ_rrt_point3d.py` 等核心模块）。
