# OmniGraph — Isaac Sim 6.0.1 官方文档

> 来源归档

- **标题：** OmniGraph — Isaac Sim Documentation
- **类型：** site（官方文档 / 教程）
- **URL（索引）：** <https://docs.isaacsim.omniverse.nvidia.com/6.0.1/omnigraph/index.html>
- **版本快照：** Isaac Sim **6.0.1**
- **文档更新：** 2026-06-22（页面 footer）
- **代码：** <https://github.com/isaac-sim/IsaacSim>
- **入库日期：** 2026-09-05
- **一句话说明：** Isaac Sim 内 OmniGraph 是 Replicator、ROS 2 bridge、传感器、控制器与外设/UI 的主编排引擎；含 Jetbot 差速控制 GUI 教程与 `omni.graph.core` Python 脚本接口。
- **沉淀到 wiki：** 是 → [`wiki/entities/omnigraph.md`](../../wiki/entities/omnigraph.md)

## 开源边界（步骤 2.5）

文档链到 [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim)（Apache-2.0）。OmniGraph 控制器与 ROS 图以 Kit 扩展与示例脚本形式随仓分发。→ **已开源**。

## 页面要点（2026-09-05）

### 在 Isaac Sim 中的角色

OmniGraph is Omniverse's visual programming framework — a graph framework connecting functions from multiple systems, with a compute backend for custom nodes.

Inside Isaac Sim, OmniGraph is the **main engine** for:

- **Replicators**（合成数据）
- **ROS 2 bridge**
- **Sensor access**
- **Controllers**
- **External input/output devices**
- **UI**

**打开编辑器：** `Window > Graph Editors > Action Graph`

### 教程索引

| 教程 | 要点 |
|------|------|
| Basic OmniGraph Tutorial | 入门 |
| **Isaac Sim OmniGraph Tutorial** | Jetbot 差速控制 Action Graph |
| **OmniGraph via Python Scripting** | `omni.graph.core` 纯脚本建图 |

### Isaac Sim OmniGraph Tutorial（Jetbot）

**学习目标：** 用 Action Graph 控制 Jetbot；熟悉差速控制器快捷方式。

**Stage 搭建：**

1. `Create > Physics > Ground Plane`
2. Content Browser：`Isaac Sim/Robots/NVIDIA/Jetbot/jetbot.usd` 拖到 stage（`/World/jetbot`）
3. Play 验证下落，Stop 后继续

**手动建图（核心节点）：**

| 节点 | 作用 |
|------|------|
| `Articulation Controller` | 对 articulation root 关节施加力/位姿/速度指令 |
| `Differential Controller` | 双轮机器人：线速度 + 角速度 → 左右轮驱动 |
| `Constant Token` ×2 | `left_wheel_joint` / `right_wheel_joint` |
| `Make Array` | `token[]` 关节名列表 → Articulation Controller |
| `On Playback Tick` | 仿真播放时每帧触发执行 |

**Differential Controller 参数（Jetbot）：** `wheelDistance=0.1125`，`wheelRadius=0.03`，`maxAngularSpeed=0.2`

**目标 Prim：** Articulation Controller 的 `robotPath` 或 `input:targetPrim` → `/World/jetbot`

### OmniGraph Shortcuts（`Tools > Robotics > OmniGraph Controllers`）

| 快捷图 | 说明 |
|--------|------|
| Joint Position Controller | 各关节位置指令 |
| Joint Velocity Controller | 各关节速度指令 |
| Differential Controller | 差速底盘；可选 WASD 键盘 |
| Open Loop Gripper Controller | 开环夹爪 |

**注意：** 不检测重复图或同机器人冲突；需自行保证场景内图唯一。快捷方式仅生成初始图，可事后编辑。

**Differential 快捷参数：** Articulation Root（如 `/World/jetbot`）、轮距、轮半径；可勾选 **Use Keyboard Control (WASD)**。生成路径默认 `/Graph/differential_controller`（冲突时自动加数字后缀）。

弹窗底部 **Python Script for Graph Generation** 可查看对应 `make_graph()` 脚本。

### Python Scripting API 摘要

模块：`import omni.graph.core as og`

| API | 用途 |
|-----|------|
| `og.Controller.edit({graph_path, evaluator_name, pipeline_stage}, {CREATE_NODES, SET_VALUES, CONNECT})` | 创建/编辑图 |
| `og.Controller.create_node(path, type)` | 增节点 |
| `og.Controller.connect(src, dst)` | 连线 |
| `og.Controller.attribute(path).get() / .set()` | 读写属性 |
| `graph_handle.evaluate()` | On-Demand 图手动求值 |
| `change_pipeline_stage(GRAPH_PIPELINE_STAGE_ONDEMAND)` | 改为按需执行 |

**示例：** `evaluator_name: "execution"`；On Tick + PrintText 每帧打印；`pipeline_stage` 设为 On Demand 时需显式 `evaluate()`。

**进阶示例路径：** `standalone_examples/api/isaacsim.core.experimental.api/omnigraph_triggers.py`（物理/渲染回调挂图）

## 对 wiki 的映射

- 实体页：[`wiki/entities/omnigraph.md`](../../wiki/entities/omnigraph.md)
- 交叉：[`wiki/entities/isaac-sim.md`](../../wiki/entities/isaac-sim.md)、[`wiki/concepts/software-in-the-loop.md`](../../wiki/concepts/software-in-the-loop.md)
- Omniverse 扩展总览：[`sources/sites/nvidia-omniverse-omnigraph.md`](./nvidia-omniverse-omnigraph.md)
