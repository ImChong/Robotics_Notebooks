# robot_descriptions.py

> 来源归档

- **标题：** robot_descriptions.py
- **类型：** repo
- **链接：** https://github.com/robot-descriptions/robot_descriptions.py
- **PyPI：** https://pypi.org/project/robot_descriptions/（包名 `robot_descriptions`，核查日版本 **3.1.0**）
- **conda-forge：** https://anaconda.org/conda-forge/robot_descriptions
- **作者 / 维护：** Stéphane Caron 等（GitHub org [`robot-descriptions`](https://github.com/robot-descriptions)）；CITATION.cff 列多名贡献者
- **许可：** 仓库 **Apache-2.0**；**各机型描述文件沿用上游许可证**（BSD / MIT / GPL / CC-BY-NC / 厂商图形条款等，README 表格逐条标注）
- **Stars：** 809（2026-08-17 核查）；forks 72
- **默认分支：** `main`（最近推送 2026-08-04）
- **入库日期：** 2026-07-30（HMI 主表节点）；**加深：** 2026-08-17
- **一句话说明：** 把 190+ 开源 URDF / MJCF 当成 Python 模块：首次 import 自动下载并缓存，再用 Pinocchio / MuJoCo / PyBullet / iDynTree / yourdfpy 等 loader 得到可直接仿真或动力学计算的模型对象。
- **开源状态：** **已开源、可运行**（`pip` / `conda` / `uvx`；CI 宣称 Awesome 列表中的条目能在对应后端加载成功）
- **策展入口：** [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)「工程与实机部署」
- **沉淀到 wiki：** [robot_descriptions.py](../../wiki/entities/robot-descriptions-py.md)
- **姊妹列表：** [awesome-robot-descriptions](awesome-robot-descriptions.md)
- **选型对照：** [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| GitHub | **已开源、可运行**：Apache-2.0；`examples/` 含各后端 load / show 脚本；PyPI + conda-forge 发版 |
| 项目页 | 无独立 `*.github.io`；文档即 README + CONTRIBUTING |
| 资产本体 | **不自托管网格**：首次 import 从各上游 git 下载到本地 cache |
| C++ 桥 | 第三方 [mayataka/robot_descriptions.cpp](https://github.com/mayataka/robot_descriptions.cpp) 调用本包 |

**边界：** 模块许可证 ≠ 机型许可证。README 对部分 UR 新机型网格标 **厂商图形文档条款 ✖️**，对 Stretch / iCub / GENE.01 等标 **CC-BY-NC 等限制条款**。复现实验前必须读对应行。

## 核心机制

1. **按名加载：** `from robot_descriptions.loaders.pinocchio import load_robot_description` → `load_robot_description("upkie_description")`。
2. **子模块路径：** `from robot_descriptions import go2_description` 暴露 `URDF_PATH` / `MJCF_PATH` / `PACKAGE_PATH` / `REPOSITORY_PATH`；部分机型另有 `URDF_PATH_POLYTOPE_COLLISION`。
3. **Xacro：** 描述可声明 `XACRO_PATH`（及 `XACRO_ARGS`），加载时透明展开成缓存 URDF。
4. **CLI：** `uvx robot_descriptions pull iiwa14_description`；`show_in_mujoco` / `show_in_meshcat` / `show_in_pybullet` / `show_in_yourdfpy`。
5. **分类：** Arms / Bipeds / Dual arms / Drones / Educational / End effectors / Humanoids / Mobile manipulators / Quadrupeds / Wheeled。后缀 `_mj_description` 多为 Menagerie 系 MJCF。

Loader 表（README）：

| 软件 | 模块 |
|------|------|
| iDynTree | `robot_descriptions.loaders.idyntree` |
| MuJoCo | `robot_descriptions.loaders.mujoco` |
| Pinocchio | `robot_descriptions.loaders.pinocchio` |
| PyBullet | `robot_descriptions.loaders.pybullet` |
| RoboMeshCat | `robot_descriptions.loaders.robomeshcat` |
| yourdfpy | `robot_descriptions.loaders.yourdfpy` |

## 与相邻目录的分工

- **发现层：** [awesome-robot-descriptions](awesome-robot-descriptions.md) 策展链接与 visual/inertia/collision 勾选；本包把其中大多数变成可 import 模块。
- **MJCF 权威资产：** [MuJoCo Menagerie](mujoco-menagerie.md) 是 `_mj_description` 的主要上游之一。
- **中国市场 ROS2 包：** [fiveages-sim/robot_descriptions](fiveages-sim-robot-descriptions.md) 覆盖 Galaxea / Agibot / EngineAI 等，本包并不一一镜像。
- **研究语料：** [URDF Files Dataset](urdf_files_dataset.md) 是冻结的解析/重复分析语料，不是运行时加载器。

## 对 wiki 的映射

- [robot_descriptions.py](../../wiki/entities/robot-descriptions-py.md)
- [awesome-robot-descriptions](../../wiki/entities/awesome-robot-descriptions.md)
- [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)
- [URDF](../../wiki/concepts/urdf-robot-description.md)
- [Pinocchio](../../wiki/entities/pinocchio.md)
- [MuJoCo](../../wiki/entities/mujoco.md)
- [Humanoid Motion Intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
