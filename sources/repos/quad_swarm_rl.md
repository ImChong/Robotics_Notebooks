# quad-swarm-rl

> 来源归档

- **标题：** quad-swarm-rl
- **类型：** repo
- **链接：** https://github.com/Zhehui-Huang/quad-swarm-rl
- **Stars：** ~0.2k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** **OpenAI Gym 兼容的多四旋翼环境**，面向群体强化学习实验（相对 gym-pybullet-drones 社区更小，作 swarm RL 补充参考）。
- **沉淀到 wiki：** [quad-swarm-rl](../../wiki/entities/quad-swarm-rl.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

提供 **多 quad 并行仿真** 与 Gym 接口，便于训练 **编队、避碰、协同追踪** 等 MARL 策略。规模与文档小于 [gym_pybullet_drones.md](gym_pybullet_drones.md)，适合作为 **第二实现** 对照 reward/观测设计。

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [gym_pybullet_drones.md](gym_pybullet_drones.md) | 功能重叠；优先维护活跃度更高的 UTIAS 环境 |
| [crazyswarm2.md](crazyswarm2.md) | 仿真 → 真机 swarm 部署链路 |
