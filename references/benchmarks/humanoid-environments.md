# 人形训练环境

人形策略训练几乎都跑在这几套环境之上。选型的分歧点不在「哪个更强」，而在 **你要并行多少环境、要不要接触精度、以及打算怎么迁到真机**。

| 环境 | 适合什么 |
|------|----------|
| [Isaac Lab](../../wiki/entities/isaac-lab.md)（原 [Isaac Gym](../../wiki/entities/isaac-gym.md)） | GPU 大规模并行采样，腿足 / 人形 RL 训练的主流选择 |
| [legged_gym](../../wiki/entities/legged-gym.md) | 腿足 RL 的经典最小实现，读代码、改奖励最快 |
| [humanoid-gym](../../wiki/entities/humanoid-gym.md) | 面向人形的训练与 sim2sim 验证流程 |
| [MuJoCo](../../wiki/entities/mujoco.md) / [MJX](../../wiki/entities/mujoco-mjx.md) | 接触解算精细，常用于验证、控制器调试与 sim2sim |
| [Isaac Sim](../../wiki/entities/isaac-sim.md) | 高保真渲染与传感器仿真，偏感知与整机场景 |
| [Genesis](../../wiki/entities/genesis-sim.md) | 较新的高速仿真栈，生态仍在成长 |

## 怎么选

- 先看 [仿真器选型指南](../../wiki/queries/simulator-selection-guide.md)；
- 纠结 MuJoCo 还是 Isaac Sim，看 [MuJoCo vs Isaac Sim](../../wiki/comparisons/mujoco-vs-isaac-sim.md)；
- 训练跑通之后，迁真机的坑集中在 [Sim2Real](../../wiki/concepts/sim2real.md) 与 [Sim2Real 上机检查单](../../wiki/queries/sim2real-checklist.md)。

## 关联页面

- [Locomotion 任务页](../../wiki/tasks/locomotion.md)
- [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
- [Isaac Lab 默认环境清单](../../wiki/entities/isaac-lab-default-environments.md)
