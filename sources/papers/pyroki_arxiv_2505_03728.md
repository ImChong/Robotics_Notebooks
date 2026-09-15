# PyRoki: A Modular Toolkit for Robot Kinematic Optimization（arXiv:2505.03728）

> 来源归档（ingest）

- **标题：** PyRoki: A Modular Toolkit for Robot Kinematic Optimization
- **类型：** paper / kinematic-optimization / inverse-kinematics / motion-retargeting / trajectory-optimization / JAX
- **arXiv abs：** <https://arxiv.org/abs/2505.03728>
- **PDF：** <https://arxiv.org/pdf/2505.03728>
- **项目页：** <https://pyroki-toolkit.github.io/>
- **代码：** <https://github.com/chungmin99/pyroki>（**已开源**；见 [repos/pyroki.md](../repos/pyroki.md)）
- **机构：** 加州大学伯克利分校（UC Berkeley）
- **会场：** IROS 2025
- **入库日期：** 2026-09-15
- **一句话说明：** 基于 **JAX** 的模块化运动学优化工具箱：用可组合运动学变量 + 代价统一 IK、轨迹优化与动捕重定向；经 **jaxls / jaxlie** 支持流形 LM 与硬约束；CPU/GPU/TPU 原生；论文报告相对 **cuRobo** 在部分 IK benchmark 上 **1.4–1.7×** 更快且误差更低。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [pyroki-toolkit.github.io](https://pyroki-toolkit.github.io/) | TL;DR、作者、BibTeX |
| 代码 | [chungmin99/pyroki](https://github.com/chungmin99/pyroki) | `pip install -e .`；examples 覆盖 IK / TO / retarget |
| 文档 | [chungmin99.github.io/pyroki](https://chungmin99.github.io/pyroki/) | API 与教程 |
| 下游 | [ProtoMotions](https://protomotions.github.io/) | v3 默认 PyRoki 批量 AMASS→机器人重定向 |
| 对照 | [cuRobo](../../wiki/entities/curobo.md) | GPU 运动生成栈；V2 论文亦与 PyRoki 做重定向约束对比 |

## 开源状态（项目页核查，2026-09-15）

- **判定：已开源。** 项目页链 arXiv；GitHub 主仓含 `src/`、`examples/`（01–14 号脚本 + hand/humanoid retarget）、`benchmark/`、文档站。
- **许可：** 仓库含 `LICENSE`（MIT 路线，以仓库为准）。
- **局限（README 自述）：** 无采样式规划器；碰撞密集场景可能慢于 cuRobo 等专用 GPU 碰撞栈；JAX JIT 对静态 shape 敏感；仅支持 revolute/continuous/prismatic/fixed 关节与球/胶囊/半空间/高度图碰撞（mesh 近似为胶囊）。

## 摘要级要点

- **问题：** 机器人运动目标多样（位姿误差、速度、碰撞、模仿演示等），现有工具往往为单任务硬编码，且跨 CPU/GPU 部署不便。
- **回答：** **PyRoki** 用 **可组合运动学变量 + 代价** 描述 IK / 轨迹优化 / 重定向，接 **非线性最小二乘**（Levenberg–Marquardt，流形与增广拉格朗日硬约束）；**JAX** 实现 **CPU / GPU / TPU** 原生优化。
- **案例：** 手部 / 人形 **motion retargeting**、在线规划与轨迹优化示例；与 **cuRobo** 等 GPU IK 库 benchmark 对比（论文摘要：1.4–1.7× 加速、更低误差——以论文表为准）。

## 核心摘录（面向 wiki 编译）

### 1) 模块能力（README 级）

| 模块 | 职责 |
|------|------|
| FK | 可微 URDF 正运动学 |
| 碰撞体 | 由 URDF 自动生成胶囊等 primitive；可微碰撞 + numpy 广播 |
| 代价库 | 末端位姿、自碰/环境碰、可操作度等；支持 autodiff 或解析 Jacobian |
| 求解器 | [jaxls](https://github.com/brentyi/jaxls) LM + 流形（[jaxlie](https://github.com/brentyi/jaxlie)）+ 硬约束增广拉格朗日 |
| 示例 | `01_basic_ik` … `14_singularity_aware_ik`；`09–12` hand/humanoid retarget |

### 2) 已知局限

- 不支持闭链 / 并联机构（仅运动树）。
- 无 RRT/PRM 等采样规划。
- 碰撞性能未宣称全面超越 cuRobo。

## 对 wiki 的映射

- 升格 [paper-notebook-pyroki.md](../../wiki/entities/paper-notebook-pyroki.md) 为完整论文实体（IK / TO / retarget 工具箱）。
- 互链：[motion-retargeting.md](../../wiki/concepts/motion-retargeting.md)、[curobo.md](../../wiki/entities/curobo.md)、[protomotions.md](../../wiki/entities/protomotions.md)。

## 参考来源

- Kim*, Yi* et al., *PyRoki: A Modular Toolkit for Robot Kinematic Optimization*, IROS 2025, [arXiv:2505.03728](https://arxiv.org/abs/2505.03728)
- [pyroki-toolkit.github.io](https://pyroki-toolkit.github.io/)
- [chungmin99/pyroki](https://github.com/chungmin99/pyroki)
