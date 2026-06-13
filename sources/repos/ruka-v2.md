# RUKA-v2

> 来源归档

- **标题：** RUKA-v2
- **类型：** repo
- **来源：** NYU / NYU Shanghai（`ruka-hand-v2` GitHub 组织）
- **链接：** <https://github.com/ruka-hand-v2/RUKA-v2>
- **License：** MIT
- **Stars / Forks：** ~7 / 0+（2026-06-13 快照）
- **入库日期：** 2026-06-13
- **最近复核：** 2026-06-13
- **一句话说明：** RUKA-v2 论文配套开源仓库：**3D 打印/CAD、装配说明、控制器、`ruka_hand` Python 包、校准与遥操作脚本**；与 v1 组织 [`ruka-hand/RUKA`](https://github.com/ruka-hand/RUKA) 分离维护。
- **沉淀到 wiki：** 是 → [`wiki/entities/ruka-v2-hand.md`](../../wiki/entities/ruka-v2-hand.md)

---

## 核心定位

**RUKA-v2** 是 NYU 团队发布的 **全栈开源腱驱动仿人灵巧手** 实现：硬件可 3D 打印复刻，软件提供 **关节空间控制、AnyTeleop 重定向、自动 motor 校准** 与 **OpenTeach 遥操作** 集成示例。项目页：<https://ruka-hand-v2.github.io/>；论文：<https://arxiv.org/abs/2603.26660>。

前代 **RUKA v1** 仓库：<https://github.com/ruka-hand/RUKA>（2025；11 主动 DoF；材料约 $1,300）。

---

## 安装（README 摘录）

```bash
git clone --recurse-submodules https://github.com/ruka-hand-v2/RUKA-v2
cd RUKA-v2
conda env create -f environment.yml
conda activate ruka_hand
pip install -r requirements.txt
pip install -e .
```

- 依赖 **Conda** 环境 `ruka_hand`；子模块需 `--recurse-submodules`（含 AnyTeleop 等第三方组件）。
- 首次使用建议先跑 README 中的 **简单连通/标定脚本** 验证电机与通信，再进入遥操作或策略训练。

---

## 仓库公开内容（截至入库日）

| 类别 | 说明 |
|------|------|
| **硬件** | 可修改 **CAD/3D 打印文件**、BOM、分步装配文档与视频（链至项目页） |
| **控制** | `ruka_hand` 包：线性 joint→motor 映射、per-motor 校准脚本 |
| **重定向** | 集成 **AnyTeleop** vector-based retargeting（人视频/关键点 → RUKA-v2 关节） |
| **传感（可选）** | 开源 **磁编码器** 3D 打印件与读取固件，用于校准与 ground truth |
| **遥操作** | **OpenTeach + Oculus VR** 示例管线（论文演示平台为 Franka 7-DoF） |
| **仿真** | URDF / 仿真 transfer 演示（项目页视频） |
| **学习** | 与 **BAKU** 行为克隆框架对接的示范任务与数据格式（见论文 §3.3） |

---

## 与 v1 的主要差异（工程读点）

| 维度 | RUKA v1 | RUKA-v2 |
|------|---------|---------|
| 主动 DoF | 11（指/拇指） | 16（指/拇指）+ 2（腕） |
| 腕 | 无 | 2-DoF 平行腕（flex/ext + radial/ulnar） |
| 指间外展 | 无 | MCP adduction/abduction（中指固定） |
| 材料成本 | ~$1,300 | ~$1,500（论文；项目页写 <$2,000） |
| 校准 | 数据驱动 tendon 模型（偏依赖动捕手套） | 解耦 retargeting + 线性映射 + **可选磁编码器** |
| 安装侧腕 | 底装为主 | **侧装** 法兰，便于桌面机械臂 |

---

## 对 wiki 的映射

- [RUKA-v2 Hand](../../wiki/entities/ruka-v2-hand.md) — 硬件/软件/学习管线归纳
- [RUKA（Paper Notebooks 待深读）](../../wiki/entities/paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md) — v1 论文索引
- [Orca Hand](../../wiki/entities/orca-hand.md) — 同类开源腱驱动对照
- [Manipulation](../../wiki/tasks/manipulation.md) — 任务语境
