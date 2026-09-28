# Cartesian_Hand

> 来源归档

- **标题：** Cartesian_Hand
- **类型：** repo
- **链接：** https://github.com/generalroboticslab/Cartesian_Hand
- **项目页：** https://generalroboticslab.com/cartesian_handv1
- **论文：** [arXiv:2609.25696](https://arxiv.org/abs/2609.25696)
- **许可证：** Apache-2.0
- **入库日期：** 2026-09-28
- **一句话说明：** Duke GRL **Cartesian Hand** 控制栈：任务编排、张量执行引擎；`studio`（真机 Feetech + viser）、`sim`（CPU MuJoCo）、`sim --warp`（GPU 批仿真）三后端。
- **代码：** https://github.com/generalroboticslab/Cartesian_Hand（**已开源**）
- **沉淀到 wiki：** [paper-cartesian-hand-linear-fingers](../../wiki/entities/paper-cartesian-hand-linear-fingers.md)

---

## 架构（README）

```text
tasks.make(name) ─┬─ Policy ─ PolicyRunner ─┬─ studio.live   servos + viser
                  └─ Task ─── TaskRunner ───┼─ sim.run       CPU MuJoCo
                                            └─ sim.run_warp  batched GPU MuJoCo
```

- **控制器** 不直接开串口；Executor 负责 I/O 与节拍；观测/命令单位为 **mm**。
- **依赖：** Python 3.10+、Linux、`torch`；`pip install -e ".[sim,studio]"`；编译 `ft_servo_ext`（C++17 + CMake）。

## 任务与物体

- `cartesian_hand/tasks/`：按机制分类（cap、scissors、triggers、pump、syringe、screwdriver、pipette、tilt 等）。
- README 注明任务文件由旧实现转录，**尚未在全部 35 物体上重跑**。

## 硬件资产

- MuJoCo：`assets/cartesian_hand/cartesian_hand.xml`
- CAD 源：`assets/cartesian_hand/source/cartesian_hand_sim.step`
- 校准/新机配置：见仓内 `docs/hardware.md`

## 对 wiki 的映射

- [paper-cartesian-hand-linear-fingers](../../wiki/entities/paper-cartesian-hand-linear-fingers.md)
- [cartesian-hand-v1-generalroboticslab.md](../sites/cartesian-hand-v1-generalroboticslab.md)
