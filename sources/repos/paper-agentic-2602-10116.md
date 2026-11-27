# SAGE 官方代码库｜NVlabs/sage

- **官方仓库：** <https://github.com/NVlabs/sage>
- **项目主页：** <https://research.nvidia.com/labs/dir/sage/>（重定向至 <https://research.nvidia.com/labs/cosmos-lab/sage/>，公开镜像页 <https://nvlabs.github.io/sage/>）
- **公开数据集：** [nvidia/SAGE-10k (Hugging Face)](https://huggingface.co/datasets/nvidia/SAGE-10k)（10,000 场景、50 种房间类型与风格、565K 个独立 3D 物体）
- **关联论文：** [SAGE: Scalable Agentic 3D Scene Generation for Embodied AI](https://arxiv.org/abs/2602.10116)
- **分支 / 许可证 / 开源状态（2026-11-27 核查）：** 默认分支 `main`；主仓库采用 **Apache-2.0** 许可证（`LICENSE`），集成并修改的第三方子目录按各自原许可证分发（`IsaacLab/` 保留 BSD-3-Clause、`M2T2/` 保留 NVIDIA License、`matfuse-sd/` 与 `robomimic/` 保留 MIT License）。**已开源**：已完整发布 MCP 客户端、MCP 场景生成服务端与 Isaac Sim 扩展、三级场景增强脚本、Isaac Lab 抓放与移动操作轨迹生成管线以及 Robomimic 策略训练入口。
- **核对范围：** 静态代码核对，未本地运行。直接核对了 `NVlabs/sage` 默认分支 `main` 的根目录 `README.md`、`LICENSE`、`client/README.md`、`server/README.md`、文件目录树以及核心脚本（`client/client_generation_robot_task.py`、`client/scripts/generate_from_robot_task.sh`、`server/layout.py`、`server/validation.py`、`server/augment/scripts/robot_data_generation_franka_arm.sh`、`server/isaaclab/data_generation_franka_pnp_type_aug_with_pose_aug.py`）。

## 仓库结构与可运行入口（来自源码与 README 核对）

- **客户端环境与入口（`client/`）：**
  - 环境配置：`client/environment.yml`、`client/pyproject.toml`、`client/uv.lock`；需配置 `client/key.json`、`client/constant.py` 与 vLLM 托管的 `Qwen/Qwen3-VL-32B-Thinking`。
  - Isaac Sim 启动与 MCP 扩展链接：将 `server/isaacsim/isaac.sim.mcp_extension` 软链至 Isaac Sim 扩展目录后，通过 `client/isaac_sim_conda.sh --no-window omni.isaac.sim --enable isaac.sim.mcp_extension` 启动后台仿真服务（`client/README.md` 注记使用 Isaac Sim 4.2.0；论文实验采用 5.0.0）。
  - 场景生成入口脚本：`client/scripts/generate_from_room_desc.sh`（单房间文本）、`client/scripts/generate_from_robot_task.sh`（调用 `client/client_generation_robot_task.py` 中的 `MCPClientOAI` 连接 `../server/layout.py`）、`client/scripts/generate_multi_rooms.sh`（多房间）、`client/scripts/generate_scene_augmentation.sh`（布局级增强）。
- **MCP 服务端与核心工具（`server/`）：**
  - `server/layout.py`（基于 `FastMCP`）：注册 `@mcp.tool()` 工具集，包括 `parse_robot_policy_requirements_for_scene_generation`、`generate_room_layout`（调用 `generate_room_structure`、`add_doors_windows`、`select_materials_for_rooms`）、`place_objects_in_room`（调用 TRELLIS 生成网格、VLM 物理属性估计、DFS/法向放置，并触发 `room_physics_critic` 与 `room_semantic_critic`）、`move_one_object_with_condition_in_room` 与 `robot_task_feasibility_correction_for_room`。
  - Isaac Sim 物理通信：`server/isaacsim/isaac_mcp/server.py`（提供 `get_isaac_connection`、`create_physics_scene`）与 `server/isaacsim/isaac.sim.mcp_extension/`。
  - 无机器人或专门场景变体：`server/layout_wo_robot.py`、`server/layout_wo_robot_multiroom.py`、`server/layout_wo_robot_scene_aug.py`、`server/validation.py`、`server/correction.py`、`server/pack_scene_to_zip.py`。
  - 后端模型服务：`server/README.md` 提供 `start_trellis_server.sh`、`Qwen/Qwen3-VL-30B-A3B-Instruct` 与 `openai/gpt-oss-120b` 的 vLLM 启动命令，以及 Objathor 门窗底图数据准备指令。
- **场景增强与机器人轨迹生成（`server/augment/` & `server/isaaclab/`）：**
  - 增强脚本：`server/augment/scripts/general_pose_augmentation.sh`、`server/augment/scripts/general_cat_augmentation.sh`，以及配套脚本 `type_aug_from_layout_with_task.py`、`type_aug_with_pose_aug_from_layout_with_task.py`。
  - 机器人数据采集：`server/augment/scripts/robot_data_generation_franka_arm.sh`、`robot_data_generation_mobile_franka.sh`、`robot_data_generation_mobile_franka_scene_aug.sh`，底层调用 `server/isaaclab/data_generation_franka_pnp_type_aug_with_pose_aug.py`、`data_generation_mobile_franka_mobile_manipulation_with_pose_aug.py`，结合 `M2T2/` 抓取预测与 `server/isaaclab/curobo_tools/curobo_planner.py`（或 `curobo_tools_mobile_grasp/`）生成 HDF5 演示数据。
- **策略训练与辅助引擎：**
  - `robomimic/`：读取生成的 HDF5 轨迹训练 Diffusion Policy。
  - `matfuse-sd/`（及 `server/floor_plan_materials/flux_generator.py`）：材质生成引擎。
  - `server/eval/`：外观评测（`eval_appearance.py`）与物理稳定性评测（`eval_physics.py`）。

## 复现边界说明

- **静态代码核对，未本地运行：** 全流程重度依赖多 GPU（论文使用 NVIDIA L40S 与多卡 vLLM 服务：client 侧 `Qwen3-VL-32B-Thinking` 文档示例为 8 卡张量并行，server 侧 `gpt-oss-120b` 示例为 4 卡张量并行，外加 TRELLIS Docker 服务、MatFuse/FLUX、Isaac Sim/Isaac Lab、M2T2、CuRobo 与 Objathor 资产下载），本地仅完成源码入口与调用链静态核对。
- **资产与环境版本差异提示：** 论文 §4.1.1 引用的 Isaac Sim 版本为 `5.0.0`，而 `client/README.md` 安装指引指向 `Isaac Sim 4.2.0` 归档包；实际部署需注意 Isaac Sim 与 Isaac Lab 扩展 API 对齐。

## 对 wiki 的映射

- [SAGE: Scalable Agentic 3D Scene Generation for Embodied AI](../../wiki/entities/paper-agentic-2602-10116.md)
- [论文原文核查](../papers/paper-agentic-2602-10116.md)
