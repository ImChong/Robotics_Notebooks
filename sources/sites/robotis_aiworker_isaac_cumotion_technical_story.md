# AI Worker × NVIDIA Isaac ROS cuMotion（ROBOTIS Technical Story）

> 来源归档（ingest）

- **标题：** AI Worker x NVIDIA Isaac ROS cuMotion
- **类型：** technical documentation / integration guide
- **URL：** <https://docs.robotis.com/docs/systems/aiworker/resources/technical_story/isaac_cumotion/>
- **视频（用户给定）：** <https://youtu.be/fmZdMV72IR0>
- **关联仓库：** [ROBOTIS-GIT/cyclo_solution](https://github.com/ROBOTIS-GIT/cyclo_solution)（`cyclo_cumotion_bringup`、`cyclo_cumotion_moveit_config` 等）
- **NVIDIA 组件：** [isaac_ros_cumotion](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion)、Nvblox、robot segmenter、object attachment
- **入库日期：** 2026-09-24
- **一句话说明：** 在 **cyclo_solution** Docker 工作站上跑 **MoveIt 2 + cuMotion**，AI Worker 侧 ZED/D405 深度与关节状态上行；演示 **四阶段**：末端目标、**静态场景**、**Nvblox ESDF 动态障碍**、**携带物体 attachment** 碰撞感知规划。

## 开源核查（2026-09-24）

| 资源 | 状态 |
|------|------|
| [cyclo_solution](https://github.com/ROBOTIS-GIT/cyclo_solution) | **已开源**（集成 launch、scene、XRDF 工作流） |
| [isaac_ros_cumotion](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion) | **已开源**（Isaac ROS 包） |
| AI Worker 当前 **JetPack 6.2** | 文档：**完整管线在外部 GPU 工作站**；上机需升级兼容 JetPack |

## 核心摘录

### 架构

- AI Worker：相机驱动 + 硬件控制器；工作站：`cyclo_solution` 内 cuMotion、robot segmenter、Nvblox。
- 数据流：ZED + 双腕 **RealSense D405** → **robot segmenter 去自影** → Nvblox **TSDF/ESDF** → RViz **MoveGroup** → `isaac_ros_cumotion_moveit` 插件 → **FollowJointTrajectory** 回 lift/双臂。

### 规划组（SRDF）

| Group | 关节 |
|-------|------|
| `arm_l` / `arm_r` | 单臂 7-DoF |
| `both_arms` | 双臂 |
| `wholebody` | **lift + 双臂** |

- **URDF + XRDF** 碰撞球在 **Isaac Sim Lula XRDF Editor** 导出；四组各配 XRDF。

### 四段演示

1. **Target pose** — `arm_l` / `arm_r` / `wholebody` 无世界碰（深度/attachment 关）。
2. **Static scene** — MoveIt `.scene`（示例 `two_open_boxes.scene` 货架），`enable_static_scene:=true`。
3. **Dynamic ESDF** — `nvblox_camera_set:=all|head|wrists`，`read_esdf_world`；障碍引入后 **重规划** 绕障（非执行中连续改轨迹）。
4. **Attached object** — `enable_object_attachment:=true`；`AttachObjectByName`（示例 `table_box`）；清除 ESDF 中已抓物体 voxel 避免双重表示。

### 工程注意

- 深度稳定：**AI Worker 后部 USB3 → USB–Ethernet → 用户 PC**。
- 局限：ESDF 仅覆盖可见表面；动态 demo 是 **replan on updated map**，非安全级 mid-execution 改轨。

## 对 wiki 的映射

- [robotis-ai-worker-isaac-cumotion](../../wiki/entities/robotis-ai-worker-isaac-cumotion.md)
- [robotis-ai-worker](../../wiki/entities/robotis-ai-worker.md)
- [Isaac ROS Nvblox](../../wiki/entities/isaac-ros-nvblox.md)
- [cuRobo](../../wiki/entities/curobo.md)
