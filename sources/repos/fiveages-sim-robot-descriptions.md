# fiveages-sim/robot_descriptions

> 来源归档

- **标题：** fiveages-sim/robot_descriptions
- **类型：** repo
- **链接：** https://github.com/fiveages-sim/robot_descriptions
- **组织：** [fiveages-sim](https://github.com/fiveages-sim)（LICENSE 版权行 `Zhenbiao@FiveAges`）
- **许可：** 仓库 **Apache-2.0**；子模块与厂商 URDF 可能另有条款；**Agibot G2 子模块为 private**
- **Stars：** 82（2026-08-17 核查）；forks 17
- **默认分支：** `main`（最近推送 2026-08-13）
- **入库日期：** 2026-08-17
- **一句话说明：** 把人形、轮式人形、四足、单臂整理成 **ROS 2 description 包**；多数在 Blender 里重绘外观；用 git submodule 拆公共手眼模型与部分厂商。
- **开源状态：** **已开源、可克隆**（需 `--recursive`）。下游 [arms_ros2_control](https://github.com/fiveages-sim/arms_ros2_control)、[robot_usds](https://github.com/fiveages-sim/robot_usds) 亦 Apache-2.0 公开。**G2 子模块未公开。**
- **沉淀到 wiki：** [fiveages-sim robot_descriptions](../../wiki/entities/fiveages-sim-robot-descriptions.md)
- **选型对照：** [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| 本仓 | **已开源**：Apache-2.0；CMake / ROS 2 包布局；README 给 `git clone --recursive` |
| 项目页 | 无独立 `*.github.io`；文档即 README |
| [arms_ros2_control](https://github.com/fiveages-sim/arms_ros2_control) | **已开源** Apache-2.0（79★，2026-08-17）；ROS 2 Control / Gazebo / OCS2 / WBC 叙事 |
| [robot_usds](https://github.com/fiveages-sim/robot_usds) | **已开源** Apache-2.0（71★）；「Isaac Sim USDs for ROS2 Control」，由本仓 URDF 转 USD |
| Agibot G2 子模块 | README 标明 **private submodule** → **部分开源**：主树公开，G2 需额外权限 |

**运行边界：** 这是 **description 资产仓**，不是训练框架。要动起来需接 `arms_ros2_control` 或 `robot_usds` + Isaac Sim。漏 `git submodule update --init` 会缺四足、common 手眼、Dobot / Tianji / Rokae / ARX / Galbot 等整枝。

## README 覆盖（主树，不含子模块细节）

| 族 | 代表机型 |
|----|----------|
| 轮式人形 | DexForce W1、Agibot G1、Airbot MMK2、Astribot S1、Galaxea R1 / R1 Pro、Realman AIDAL / RS-01、Zerith H1、Ai2 Bot2、XSquare Quanta X1、Spirit AI MOZ 1 |
| 移动操作 | Galaxea R1 Lite、AgileX Aloha、SIGRobotics Lekiwi |
| 单臂 | SO-ARM、AgileX Piper、Galaxea A1 系列、Airbot Play、Realman RM65/75、Elite EC、OpenArm、HighTorque Panthera HT |
| 双足人形 | Unitree G1、Agibot A2、Booster T1、EngineAI SA01 / PM01、RobotEra xbot |

子模块（README 表）：`common`（手/夹爪/相机/launch）、`quadruped`、Dobot CR5、Tianji M6、Rokae AR5、ARX、Galbot、Agibot G2（private）、Panthera HT。

## 与 robot_descriptions.py 的差异

- **生态：** ROS 2 包 + Isaac USD 姊妹仓，不是 Python lazy-download。
- **机型重心：** 近年国内轮式人形 / 协作臂 / 开源臂（Piper、SO-ARM）更全；国际学术经典机（ANYmal、TALOS、iCub）应走 Awesome / `robot_descriptions.py`。
- **外观：** 多数「Repaint = Yes」，可视化友好，**不等于惯量/碰撞已标定到真机**。

## 对 wiki 的映射

- [fiveages-sim robot_descriptions](../../wiki/entities/fiveages-sim-robot-descriptions.md)
- [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)
- [URDF](../../wiki/concepts/urdf-robot-description.md)
- [Isaac Sim](../../wiki/entities/isaac-sim.md)
- [robot_descriptions.py](../../wiki/entities/robot-descriptions-py.md)
