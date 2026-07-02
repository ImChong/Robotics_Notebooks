---
type: entity
tags: [repo, manipulation, rope, iterative-learning-control, simulation, cmu, open-source]
status: complete
updated: 2026-07-02
related:
  - ./paper-flying-knots.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/repos/flying_knots_public.md
summary: "flying_knots_public 是 Flying Knots（arXiv:2602.21302）MIT 开源研究代码：Vicon 示教、粒子绳仿真、逆模型 QP ILC 与 xArm7 真机接口的四阶段管线。"
---

# flying_knots_public（开源仓库）

**flying_knots_public** 是 CMU Flying Knots 论文的 **MIT 许可研究代码快照**，实现 **示教采集 → 清洗标注 → IK 初始命令 → Task-Level ILC** 全链路。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ILC | Iterative Learning Control | 仓库核心算法循环在 `main/learning.py` |
| QP | Quadratic Programming | 逆模型更新在 `simulation/inverse_model.py` |
| IK | Inverse Kinematics | 初始命令由 `main/compute_ik.py` + Drake SNOPT 生成 |
| URDF | Unified Robot Description Format | xArm7 模型在 `models/xarm_description/` |
| MoCap | Motion Capture | Vicon 客户端在 `common/mocap.py` |

## 核心信息

| 字段 | 内容 |
|------|------|
| 链接 | <https://github.com/krish-suresh/flying_knots_public> |
| 论文 | [Flying Knots](./paper-flying-knots.md)（arXiv:2602.21302） |
| 依赖管理 | `uv sync` |
| 数据根目录 | `$FLYING_KNOT_DATA`（默认 `~/flying_knot_data`） |

## 为什么重要

- **paper→code 可追踪**：`docs/architecture.md` 给出 Algorithm 1、Eq. 3/4 与源文件对照。
- **仿真/硬件双路径**：ILC 可在粒子仿真或 xArm7+Vicon 真机上跑同一 `learning.py` 循环。
- **模块化绳模型**：粒子主路径 + Drake/Elastica 备选，便于 ablation 与可视化（Viser）。

## 快速入口

```bash
git clone https://github.com/krish-suresh/flying_knots_public
cd flying_knots_public && uv sync
# 配置 config/hardware、config/learning 后按 main/ 脚本顺序运行
```

## 关联页面

- [Flying Knots（论文实体）](./paper-flying-knots.md)
- [Manipulation（操作）](../tasks/manipulation.md)

## 参考来源

- [flying_knots_public 仓库归档](../../sources/repos/flying_knots_public.md)
- Suresh & Atkeson, arXiv:2602.21302 — 算法与实验细节以论文为准

## 推荐继续阅读

- 仓库 `docs/architecture.md` — 模块与数据布局
- [Flying Knots 项目页](https://flying-knots.github.io/)
