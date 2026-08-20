# HorizonRobotics / RoboOrchardLab

> 来源归档

- **标题：** RoboOrchardLab
- **类型：** repo（Python 包 + 算法项目集合）
- **维护方：** Horizon Robotics（地平线）
- **链接：** <https://github.com/HorizonRobotics/RoboOrchardLab>
- **文档站：** <https://horizonrobotics.github.io/robot_lab/robo_orchard/lab/index.html>
- **PyPI：** `robo_orchard_lab`（**0.5.0** 基线，2026-08-20）
- **许可：** Apache-2.0
- **入库日期：** 2026-08-20
- **GitHub：** 约 **163** stars（2026-08-20）
- **一句话说明：** 地平线 **RoboOrchard** 生态下的 **模块化具身 AI 训练与评测框架**：核心包 `robo_orchard_lab` 提供可插拔训练管线（Hook / Trainer / Pipeline），与 Hugging Face **Accelerate / Datasets** 生态对齐，并在 `projects/` 收纳 **HoloBrain、FineGrasp、BIP3D、PickPlaceAgent** 等 SOTA 算法实现。
- **沉淀到 wiki：** [`wiki/entities/robo-orchard-lab.md`](../../wiki/entities/robo-orchard-lab.md)
- **文档站归档：** [`sources/sites/robo-orchard-lab-docs.md`](../sites/robo-orchard-lab-docs.md)

---

## 与「robot_lab」命名的关系

文档站点路径含 `robot_lab/robo_orchard`，指 **Horizon Robotics 组织下托管的 GitHub Pages 文档树**，与社区维护的 IsaacLab 扩展 **[fan-ziqi/robot_lab](robot_lab.md)** 不是同一项目；阅读文档与引用链接时建议 **以组织名与域名区分**。

---

## 仓库结构（README / 文档索引，2026-08-20）

| 路径 | 职责 |
|------|------|
| `robo_orchard_lab/` | 核心 Python 包：dataset、pipeline、trainer、models、metrics、distributed 等 |
| `robo_orchard_lab/pipeline/` | `trainer.py`、`hook_based_trainer.py`、训练 Hook 与推理入口 |
| `projects/holobrain/` | HoloBrain-0 通用操作 VLA 基础模型 |
| `projects/finegrasp_graspnet1b/` | FineGrasp 精细抓取检测 |
| `projects/bip3d_grounding/` | BIP3D 2D–3D 具身感知（CVPR 2025） |
| `projects/pick_place_agent/` | PickPlaceAgent（RoboTwin 2.0 评测示例） |
| `projects/mapdream/` · `projects/monodream/` | VLN 地图 / 单目导航 |
| `projects/aux_think/` · `projects/progress_think/` · `projects/sem/` | 辅助推理、进度思考、SEM 策略等 |
| `docs/` | Sphinx 文档（安装、Dataset / Trainer / Model Zoo 教程） |
| `examples/` | 数据集与 ResNet50 ImageNet 示例 |
| `orchard_config.toml` | 构建依赖声明（`robo_orchard_core`） |

## 安装要点（官方文档）

- **环境：** Linux（Ubuntu 22.04 已测）、Python **≥ 3.10**、PyTorch **≥ 2.4.0**（需先手动安装匹配 CUDA 的 torch）
- **PyPI：** `pip install robo_orchard_lab`；算法可选 extra：`[holobrain_0]`、`[finegrasp]`、`[bip3d]`、`[sem]`、`[aux_think]`、`[mcap_datasets]` 等
- **源码：** `make version && make install-editable`（或 `pip install -e .`）
- **开发：** `make dev-env`（pre-commit、lint、测试依赖）

## 资料在知识库中的角色

| 资料 | 角色 |
|------|------|
| 本文件 | 官方入口（代码 / 文档 / PyPI / 算法项目）一站式索引 |
| [robo-orchard-lab-docs.md](../sites/robo-orchard-lab-docs.md) | 文档站与开源状态核查 |
| [horizon_robotics_holomotion.md](horizon_robotics_holomotion.md) | 同组织人形运动跟踪栈（HoloMotion，不同子项目） |
| [holoagent.md](holoagent.md) | 同组织 Agent / 导航栈（HoloAgent） |
