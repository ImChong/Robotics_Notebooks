# ntnu-arl/hflex_eqa（HFLEX-EQA 官方源码归档）

- **官方仓库：** <https://github.com/ntnu-arl/hflex_eqa>（主库，默认分支 `main`，BSD-3-Clause）
- **配套 ROS 2 仓库：** <https://github.com/ntnu-arl/hflex_eqa_ros>（ROS 2 Jazzy 节点、Launch、RViz 面板与仿真管理，默认分支 `main`，BSD-3-Clause）
- **项目主页：** <https://ntnu-arl.github.io/hflexeqa/>
- **对应论文：** `arXiv:2609.26360`（[wiki 实体页](../../wiki/entities/paper-agentic-2609-26360.md)）
- **核查边界：** 静态代码与文档核对（基于公开仓库 README、文件列表与依赖清单核查），**未在本地构建 Docker 或运行实机/Habitat 仿真**。

## 仓库结构与关键模块

### 1. 主仓库 `ntnu-arl/hflex_eqa`
- **`hflex_eqa/`（C++ 规划核心库，colcon 包）：**
  - 核心流水线：`hflex_eqa/include/hflex_eqa/common/hflex_eqa_pipeline.h`、`hflex_eqa/src/common/hflex_eqa_pipeline.cpp`
  - 搜索管理与高层调度：`hflex_eqa/include/hflex_eqa/search_manager/eqa_planner_manager_module.h`、`baseline_manager_module.h`、`search_manager_module.h`
  - 底层规划模块（对应三种模式与路径规划）：
    - `hflex_eqa/include/hflex_eqa/low_level/low_level_module.h` 及 `src/low_level/low_level_module.cpp`
    - `find_room_search.h/.cpp`（`find_room` 平面图拓扑加权与入口语义搜索）
    - `object_vps_search.h/.cpp`（`go_to_objects` 可见性视点采样与评估）
    - `question_room_clip_search.h/.cpp`、`semantic_search.h/.cpp`、`clip_search.h/.cpp`（`explore_room` 语义前沿匹配）
    - `a_star.h/.cpp`、`dijkstra.h/.cpp`、`bspline_planner.h/.cpp`、`cubic_spline_planner.h/.cpp`
  - 配置与提示词：`hflex_eqa/config/habitat/`（`habitat.yaml`、`high_level.yaml`、`eqa_planner_with_floorplan_system_prompt.txt` 等）与 `hflex_eqa/config/anymal/`（`high_level.yaml`、`low_level.yaml` 等）
  - 评测脚本：`hflex_eqa/eval/scripts/evaluate_eqa_results.py`、`evaluate_grapheqa_results.py`
- **`hflex_eqa_python/`：** VLM 规划器与评测辅助工具。
- **`docker/` 与 `install/`：** `docker/default/`（桌面 CUDA + Habitat）、`docker/thor/`（NVIDIA Jetson Thor 实机镜像），以及 `install/default.repos`、`install/thor.repos`。

### 2. 配套 ROS 2 仓库 `ntnu-arl/hflex_eqa_ros`
- **`hflex_eqa_ros/`（C++/Python ROS 2 封装）：**
  - 可执行入口节点：`hflex_eqa_ros/app/hflex_eqa_node.cpp`（C++ 规划流水线节点）、`hflex_eqa_ros/app/vlms_node.py`（高阶 VLM 推理与任务解析节点）
  - 场景图接收与接口：`hflex_eqa_ros/src/hflex_eqa_ros_pipeline.cpp`、`hflex_eqa_ros/hflex_eqa_ros/dsg_ros_receiver.py`、`hflex_eqa_ros/src/io/graph_ros_wrapper.cpp` / `graph_zmq_wrapper.cpp`
  - Launch 启动入口：`hflex_eqa_ros/launch/habitat/habitat_eqa.launch.yaml`、`habitat_high_level.launch.yaml`、`habitat_hflex_eqa.launch.yaml`、`habitat_search_manager.launch.yaml`；实机通用入口：`hflex_eqa_ros/launch/scene_graph.launch.yaml`、`hflex_eqa_ros/launch/eqa.launch.yaml`
- **`hflex_eqa_msgs/` & `hflex_eqa_ui/`：** 自定义消息（`EQAPlannerOutput.msg`、`HighLevelPlannerOutput.msg`、`ObjectsViews.msg` 等）与 RViz 交互插件 `eqa_panel.cpp`。
- **`simulation_manager_ros/`：** 基准批测入口 `launch/simulate.launch.yaml`、`app/habitat_eqa_manager_node.py`，以及 HM3D 拓扑平面图生成脚本 `data/hm3d/generate_floorplans.py`、`generate_floorplans_exploreeqa.py`。

## 运行依赖与复现边界

- **系统环境：** Ubuntu 24.04 / ROS 2 Jazzy，Docker Compose v2 + NVIDIA GPU（支持桌面仿真与 Jetson Thor 部署）。
- **外部感知与地图依赖：** 需通过 `vcstool` 拉取 `reasoning_hydra`、`semantic_inference_ros`、`habitat-ros`、`kimera_pgmo`、`kimera_rpgo`、`vlms_ros` 等依赖分支；需手动准备 `models/lseg.ckpt`，并在联网首次运行时下载 `yoloe-11l-seg.pt` 与 OpenCLIP ViT-B/32 缓存（8 GB 显存可设 `HFLEX_EQA_QUERY_DEVICE=cpu`）。
- **数据与 API 边界：** 仿真评测需申请 HM3D v0.2 语义数据集及统计文件 `Per_Scene_Region_Weighted_Votes.csv` 并预生成 `topological_graph.json`；高阶 VLM 规划支持 OpenAI、Gemini、Llama、Anthropic 或本地 vLLM OpenAI 兼容接口（需在 `.env` 配置对应 API Key）。
