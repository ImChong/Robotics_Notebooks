# Habitat Sim Docs（aihabitat.org/docs/habitat-sim）

> 来源归档（site · Habitat-Sim 官方文档）

- **标题：** Habitat Simulator | Habitat Sim Docs
- **类型：** site / documentation
- **URL：** <https://aihabitat.org/docs/habitat-sim/>
- **配套仓库：** <https://github.com/facebookresearch/habitat-sim> — [`sources/repos/habitat-sim.md`](../repos/habitat-sim.md)
- **门户：** <https://aihabitat.org/> — [`sources/sites/aihabitat-org.md`](aihabitat-org.md)
- **Lab Docs：** <https://aihabitat.org/docs/habitat-lab/>
- **入库日期：** 2026-08-04
- **一句话说明：** Habitat-Sim 的 Python 优先 API 教程与类参考：导航 / 交互 / 刚体 / 光照 / 立体相机 / NavMesh / 传感器等；C++ 内部另有 API 页。

## 文档宣称的核心指标

与 README 一致：Matterport3D 场景单线程数千 FPS，单 GPU 多进程 **>10,000 FPS**；平台由 **Habitat-Sim + Habitat-Lab** 组成，协作关系见 ECCV 2020 tutorial series。

## 教程与主题索引（页首导航）

| 类别 | 主题示例 |
|------|----------|
| Basics | Navigation、Interaction、Advanced Topics、Profiling（含 Video / Jupyter / Colab） |
| Config | New Actions、Attributes Templates JSON、Stereo agent、Light setups |
| Assets | View Assets、Interactive Rigid Objects 2.0、Gfx Replay、Blender 编辑、Coordinate Frame |
| API | Python Classes 标签页、Logging Configuration、精选 unit tests（`test_agent.py`、`test_physics.py`、`test_sensors.py`、`test_navmesh.py` 等） |
| C++ | 内部 API 标签（面向贡献者；终端用户以 Python 为主） |

## 工程读法

1. **先跑 Navigation / Interaction notebook**，再接 Lab 任务配置。
2. **物理路径**需确认安装变体含 Bullet（`withbullet`）。
3. **场景许可**不在本 docs 页自动解决——按 Datasets 页申请 MP3D/HM3D 等。
4. Meta 已声明 **v0.3.4 之后不再官方主动维护**（见仓库 README）；文档仍是 API 真相来源，但 issue/feature 节奏按社区预期。

## 对 wiki 的映射

- [`wiki/entities/habitat-sim.md`](../../wiki/entities/habitat-sim.md) — 「工程实践」与运行时序图对齐本页入口
- [`sources/repos/habitat-sim.md`](../repos/habitat-sim.md)
