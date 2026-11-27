# embodied-nav-mcp（AnchorVLN 官方 MCP 服务与挑战赛运行仓）

- **官方仓库：** https://github.com/aryanmangal769/embodied-nav-mcp
- **关联论文：** [AnchorVLN: Geometry-Anchored Vision-Language Grounding Reasoning for Open-Vocabulary Navigation](../../sources/papers/paper-agentic-2609-12285.md)（`arXiv:2609.12285`）
- **开源协议：** MIT（底层几何与 ROS I/O 代码源自 Team Xiao Hei 仓库 [ginlov/CMU-VLN-Challenge-2026](https://github.com/ginlov/CMU-VLN-Challenge-2026) 的 `feat/xiao-hei-instruction-merge` 分支，协议说明见仓库 `NOTICE`）
- **默认分支：** `main`
- **核对范围：** 仓库根目录结构、`src/embodied_mcp/` 核心模块清单、`examples/cmu_vln/` ROS 2 适配层文件列表、`README.md` 与 `tests/test_tools.py` 清单（静态代码与文档核对，未本地运行）
- **复现边界：** 静态代码核对，未本地运行；实机/仿真运行依赖 ROS Jazzy 与 CMU VLN 自主导航栈，仓库内置 `--fake` 干跑与 `pytest` 单元测试不依赖 ROS

## 仓库结构与入口核对

### 1. 核心 MCP 服务与智能体模块（`src/embodied_mcp/`）

- `src/embodied_mcp/server.py`：MCP 工具服务端实现（支持 `--attach-ros` 通过 stdio MCP 暴露工具）
- `src/embodied_mcp/client.py`：VLM 代理客户端驱动（默认模型环境变量 `EMBODIED_MCP_MODEL=claude-opus-5`，依赖 `ANTHROPIC_API_KEY`）
- `src/embodied_mcp/tools.py`：工具模式与约束封装（不接收米制距离或弧度方位角数值参数）
- `src/embodied_mcp/challenge.py`：挑战赛进程内直连入口（`embodied-mcp-challenge`），规避机器人容器无公网 URL 的远程连接限制
- `src/embodied_mcp/session.py`、`src/embodied_mcp/memory.py`、`src/embodied_mcp/result.py`：单题会话状态、句柄/历史记录及结果发布结构
- `src/embodied_mcp/vendor_path.py`：加载内置的 `_vendor/xiao_hei_vlm/` 感知与控制预测组件

### 2. 内置几何与 ROS I/O 依赖（`src/embodied_mcp/_vendor/xiao_hei_vlm/`）

- 几何感知与尺寸先验：`perception/geometry.py`、`src/xiao_hei_vln/perception/geometry.py`、`src/xiao_hei_vln/perception/size_prior.py`（实现尺寸门控、盲区锥检测与 LiDAR 测距提升）
- 控制器预测与运动规划：`scripts/waypoint_converter_model.py`（路点转换器停靠点预测模型）、`scripts/approach_loop.py`、`scripts/execute_plan.py`、`scripts/vlm_locate.py`、`scripts/vlm_approach.py`
- 计数与视图工具：`scripts/faces.py`、`scripts/grab_faces.py`、`scripts/count_view.py`、`scripts/answer_numerical.py`、`scripts/robot_io.py`、`robot_node.py`

### 3. CMU VLN Challenge 2026 ROS 2 适配层（`examples/cmu_vln/`）

- `examples/cmu_vln/dummy_vlm/launch/dummy_vlm.launch`：官方评测器调用入口（`ros2 launch dummy_vlm dummy_vlm.launch`）
- `examples/cmu_vln/dummy_vlm/src/dummyVLM.py`：保留 `dummy_vlm` 包名的即插即用节点包装器
- `examples/cmu_vln/Dockerfile.overlay`：挑战赛容器叠加构建文件

### 4. CLI 与 ROS 话题契约

- **离线干跑：** `uv run embodied-mcp-agent --fake "<instruction>"`
- **单元测试：** `uv run pytest`（使用 `tests/test_tools.py` 中的模拟机器人，无需 ROS）
- **订阅传感器话题（挑战赛允许白名单）：** `/camera/image`（可通过 `XIAO_HEI_IMAGE_TOPIC` 配置）、`/registered_scan`、`/terrain_map`、`/terrain_map_ext`、`/state_estimation`（用于直接判定到达而非信任 `/way_point_reached`）
- **发布输出话题：**
  - 指令跟随：`get_view` / `ground` / `drive_to` / `stop` -> `/way_point_with_heading`
  - 物体指代：`ground` / `publish_object` -> `/selected_object_marker`
  - 数值计数：`parse_numerical` / `count_in_view` / `commit_count` -> `/numerical_response`
- **预算与运行时限制：** 默认单题进程预算 `EMBODIED_MCP_BUDGET_S=540` 秒，每进程单题评测。
