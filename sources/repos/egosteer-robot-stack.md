# egosteer/robot-stack

> 来源归档

- **标题：** EgoSteer Unified Robot Stack（遥操作 / 推理 / HITL 纠偏）
- **类型：** repo
- **组织 / 作者：** egosteer（PKU / PKU–PsiBot）
- **代码：** <https://github.com/egosteer/robot-stack>
- **镜像：** `egosteerai/robot-stack:latest`（亦有 GHCR / `docker-registry.psibot.net`）
- **硬件默认：** RealMan 双臂 + RuiYan RY-H2 双手 + PsiBot SynGlove-Air + Vive Trackers + RealSense 头/胸相机
- **论文：** <https://arxiv.org/abs/2607.09701>
- **项目页：** <https://egosteer.github.io/>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-23
- **一句话说明：** ROS 2 Humble Docker 栈；同一低层控制节点服务遥操作采集、策略推理客户端与脚踏 HITL DAgger，相对运动映射保证接管平滑。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `./create_container.sh robot-stack` | 创建并进入挂载仓的 Docker 环境 |
| `steamvr` / `robot --teleop` / `collect` | 遥操作采集闭环 |
| `robot --inference` / `interface` | 连 EgoSteer WebSocket 服务做真机推理 |
| `robot --replay` / `replay` | 回放 `.rrd` 轨迹 |
| `assets/udev_rules/install.sh` | 固定手/手套串口设备名 |
| `src/tracker/config/dual_tracker.yaml` | Vive tracker 序列号 |

## 与本仓库知识的关系

- 论文归档：[`sources/papers/egosteer_arxiv_2607_09701.md`](../papers/egosteer_arxiv_2607_09701.md)
- 策略服务端：[`egosteer`](./egosteer.md)
- wiki：[`wiki/entities/paper-egosteer.md`](../../wiki/entities/paper-egosteer.md) · [`wiki/methods/dagger.md`](../../wiki/methods/dagger.md) · [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)
