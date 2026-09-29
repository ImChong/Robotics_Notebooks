# pycarm

> 来源归档

- **标题：** pycarm
- **类型：** repo（SDK）
- **机构：** 视源股份（CVTE Robotics）
- **链接：** https://github.com/cvte-robotics/pycarm
- **描述：** Python interface for cvte arm.
- **星标（截至 2026-09-29）：** ~1
- **入库日期：** 2026-09-29
- **一句话说明：** CARM 机械臂 Python SDK；[`carm-lerobot`](carm_lerobot.md) 真机驱动通过 `import carm` / `CArmSingleCol` 接臂。
- **沉淀到 wiki：** 否（由 [`carm-lerobot` 实体页](../../wiki/entities/carm-lerobot.md) 引用）

---

## 与 LeRobot 改版的关系

- [`carm-lerobot`](carm_lerobot.md) README 要求：`pip install carm`（PyPI 包名 **carm**，源码仓为本仓库）。
- A3 从臂在 `src/lerobot/robots/carm_a3/robot_a3.py` 中实例化 `carm.CArmSingleCol(addr)` 并完成 `set_ready()`、关节/夹爪初始化。

## 开源状态

- **已开源：** 公开 GitHub 仓库；许可证字段在 API 未标注 SPDX，以仓内 LICENSE 为准（入库日未逐文件核验）。

## 对 wiki 的映射

- 见 [`wiki/entities/carm-lerobot.md`](../../wiki/entities/carm-lerobot.md) 工程实践表。
