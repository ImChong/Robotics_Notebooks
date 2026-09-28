# lerobot-legged-zoo

> 来源归档

- **标题：** lerobot-legged-zoo
- **类型：** repo
- **链接：** https://github.com/Virgileboat/lerobot-legged-zoo
- **入库日期：** 2026-09-28
- **一句话说明：** 多足/人形 **MJCF 模型 + MJLab 训练示例** 合集；含 LeRobot Humanoid 平地/崎岖 **速度跟踪**任务（`uv run train/play`），**不提供预训练策略**。
- **代码：** https://github.com/Virgileboat/lerobot-legged-zoo（**已开源**；license 以仓内文件为准）
- **沉淀到 wiki：** [lerobot-humanoid](../../wiki/entities/lerobot-humanoid.md)

---

## LeRobot Humanoid 训练任务（README 表）

| Task | 说明 |
|------|------|
| `Mjlab-Velocity-Flat-LeRobot-Humanoid` | 平地行走 |
| `Mjlab-Velocity-Rough-LeRobot-Humanoid` | 崎岖地形 |
| `Mjlab-Velocity-Flat-LeRobot-Humanoid-full` | full 变体平地 |
| `Mjlab-Velocity-Rough-LeRobot-Humanoid-full` | full 变体崎岖 |

同仓亦含 Open Duck v2、Unitree G1（23/29 DoF）、Leggy 等任务。

## 环境与命令

```bash
uv sync
uv run train Mjlab-Velocity-Flat-LeRobot-Humanoid --env.scene.num-envs 2048
uv run play Mjlab-Velocity-Flat-LeRobot-Humanoid --checkpoint-file logs/.../model_*.pt
```

- **需要 CUDA GPU** 进行训练/play（README）

## 状态

- LeRobot Humanoid 族：**in development**；CAD 与 hardware 仓 Onshape 对齐

## 对 wiki 的映射

- [LeRobot Humanoid](../../wiki/entities/lerobot-humanoid.md)
