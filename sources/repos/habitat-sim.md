# Habitat-Sim（facebookresearch/habitat-sim）

> 来源归档（repo · Meta AI Habitat 仿真核心）

- **标题：** Habitat-Sim — A flexible, high-performance 3D simulator for Embodied AI research
- **类型：** repo
- **来源：** Meta Platforms / facebookresearch（原 FAIR）
- **链接：** <https://github.com/facebookresearch/habitat-sim>
- **主页：** <https://aihabitat.org/> — 归档见 [`sources/sites/aihabitat-org.md`](../sites/aihabitat-org.md)
- **文档：** <https://aihabitat.org/docs/habitat-sim/> — 归档见 [`sources/sites/aihabitat-habitat-sim-docs.md`](../sites/aihabitat-habitat-sim-docs.md)
- **许可证：** MIT
- **配套高层库：** [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)（任务 / 训练 / 评测；另仓）
- **入库日期：** 2026-08-04
- **一句话说明：** 具身 AI 高速 3D 仿真器：真实扫描与 CAD 场景、可配置 RGB-D/本体传感器、URDF 机器人 + Bullet 刚体；设计哲学是**吞吐优先于仿真能力广度**。
- **沉淀到 wiki：** 是 → [`wiki/entities/habitat-sim.md`](../../wiki/entities/habitat-sim.md)

## 开源状态（步骤 2.5，截至 2026-08-04）

| 资源 | 状态 |
|------|------|
| 训练/仿真核心代码 | **已开源**（MIT；conda / pip / Docker / 源码四种安装） |
| 文档与教程 | **已发布**（aihabitat.org Sim Docs + ECCV 2020 教程系列） |
| 场景数据集 | **部分需申请**（MP3D / Gibson / HM3D 等各有许可；ReplicaCAD / HSSD 等见官网 Datasets） |
| Meta 内部维护 | README 警告：**Beyond v0.3.4 起 Meta 内部团队不再做官方主动开发/维护**；欢迎社区 fork |

**结论：确认已开源（MIT）。** 选型时须把「维护进入社区阶段」与「代码仍可安装」分开读。

## 仓库与能力要点（README 核对）

- **场景：** HM3D、HSSD、Matterport3D、Gibson、Replica；CAD / 分件刚体如 ReplicaCAD、YCB、Google Scanned Objects。
- **传感器：** RGB-D、egomotion 等可配置。
- **机器人：** URDF（Fetch、Franka、AlienGo 等）。
- **物理：** Bullet 刚体（conda 常用 `withbullet` 变体）。
- **吞吐（官方宣称）：** MP3D 场景单线程数千 FPS、单 GPU 多进程 **>10,000 FPS**；Fetch@ReplicaCAD（128×128 RGBD + 1/30 s 刚体）**>8,000 SPS**。
- **安装（推荐）：** `conda install habitat-sim withbullet -c conda-forge -c aihabitat`（headless / 显示机参数可链式组合）；Python ≥3.9。
- **入口示例：** `examples/example.py`、`examples/demo_runner.py`、`examples/tutorials/`；单元测试（`test_sensors.py`、`test_physics.py`、`test_navmesh.py` 等）亦作 API 示范。
- **最新 GitHub Release（核查时）：** v0.3.3（2026-02-12）；`main` 仍有社区/残余推送（如 2026-07）。

## 与 Habitat-Lab 的分工

| 层 | 仓库 | 职责 |
|----|------|------|
| Sim | `habitat-sim` | 渲染、传感器、物理步进、场景/资产加载 |
| Lab | `habitat-lab` | 任务定义（Nav / Rearrange / 指令跟随等）、训练与标准指标评测 |

典型研究路径：Sim 提供 `habitat_sim.Simulator` 观测循环 → Lab 叠 Gym/任务与 baseline。

## 引用（README）

官方要求使用平台时引用 Habitat 1.0 / 2.0 / 3.0：

- Habitat 1.0（ICCV 2019）：arXiv:[1904.01201](https://arxiv.org/abs/1904.01201)
- Habitat 2.0（NeurIPS 2021）：arXiv:[2106.14405](https://arxiv.org/abs/2106.14405)
- Habitat 3.0：arXiv:[2310.13724](https://arxiv.org/abs/2310.13724)

## 对 wiki 的映射

- 加深实体 [`wiki/entities/habitat-sim.md`](../../wiki/entities/habitat-sim.md)
- 对照十年地图 [`wiki/overview/sim-platforms-decade-technology-map.md`](../../wiki/overview/sim-platforms-decade-technology-map.md)
- VLN / ObjectNav 宿主语境见 [`wiki/tasks/vision-language-navigation.md`](../../wiki/tasks/vision-language-navigation.md)、[`wiki/entities/paper-zonda.md`](../../wiki/entities/paper-zonda.md)
