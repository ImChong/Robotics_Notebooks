# flying_knots_public（仓库）

> 来源归档（ingest；论文策展见 `sources/papers/flying_knots_arxiv_2602_21302.md`）

- **标题：** flying_knots_public — Task-Level ILC for Dynamic Rope Manipulation
- **类型：** repo
- **链接：** <https://github.com/krish-suresh/flying_knots_public>
- **论文：** arXiv:2602.21302
- **项目页：** <https://flying-knots.github.io/>
- **机构：** Carnegie Mellon University
- **入库日期：** 2026-07-02
- **许可证：** MIT
- **一句话说明：** Flying Knots 论文 **研究代码快照**：四阶段管线（Vicon 示教 → 清洗/标注 critical point → Drake IK 初始命令 → Task-Level ILC 真机/仿真迭代），含粒子绳动力学、逆模型 QP、xArm7 接口与 Viser 可视化。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-flying-knots.md`](../../wiki/entities/paper-flying-knots.md)

## 仓库结构（摘要）

| 目录 | 作用 |
|------|------|
| `main/` | `human_capture`、`clean_demo`、`compute_ik`、`learning` 等工作流入口 |
| `simulation/` | 粒子/Drake/Elastica 绳模型与 `inverse_model.py`（Eq. 3 QP） |
| `xarm7/` | Drake 运动学、真机 HTTP 轨迹下发与关节 socket 采集 |
| `common/` | 数据类、Bézier 数学、QP 求解器、mocap 跟踪 |
| `config/` | 硬件、学习、仿真、IK YAML |
| `docs/architecture.md` | 模块说明与 paper→code 对照表 |

## 运行前提（仓库自述）

- 依赖管理：`uv sync`
- 数据目录：环境变量 `FLYING_KNOT_DATA`（默认 `~/flying_knot_data`）
- 真机执行需 **xArm7** 与 **Vicon**（未随仓库提供）

## 对 wiki 的映射

- [`wiki/entities/paper-flying-knots.md`](../../wiki/entities/paper-flying-knots.md) — 方法、实验与代码入口归纳
