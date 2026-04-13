# Sources

这里是 `Robotics_Notebooks` 的原始资料层。

目标不是直接回答问题，而是作为知识库的输入来源。

## 当前资料文件

### papers/ — 论文来源归档
| 文件 | 内容 |
|------|------|
| [locomotion_rl.md](papers/locomotion_rl.md) | 人形/腿足机器人 RL 论文归档 |
| [sim2real.md](papers/sim2real.md) | Sim2Real 域随机化/系统辨识论文归档 |
| [survey_papers.md](papers/survey_papers.md) | 综述论文归档 |
| [imitation_learning.md](papers/imitation_learning.md) | 模仿学习论文归档 |
| [whole_body_control.md](papers/whole_body_control.md) | 全身控制/TSID/QP-WBC 论文归档 |
| [humanoid_hardware.md](papers/humanoid_hardware.md) | 人形机器人硬件论文归档 |

### repos/ — 代码仓库来源归档
| 文件 | 内容 |
|------|------|
| [mujoco.md](repos/mujoco.md) | MuJoCo 物理引擎 |
| [isaac_gym_isaac_lab.md](repos/isaac_gym_isaac_lab.md) | Isaac Gym / Isaac Lab |
| [pinocchio.md](repos/pinocchio.md) | Pinocchio 动力学库 |
| [crocoddyl.md](repos/crocoddyl.md) | Crocoddyl 最优控制框架 |
| [unitree.md](repos/unitree.md) | Unitree 硬件与 SDK |
| [legged_gym.md](repos/legged_gym.md) | legged_gym 训练框架 |

### blogs/ — 博客来源归档
（待填充）

### notes/ — 原始笔记归档
| 文件 | 内容 |
|------|------|
| [know-how.md](notes/know-how.md) | 人形机器人技术框架、Know-How 文档、深蓝学院课程 |
| [legacy-readme-resource-map.md](notes/legacy-readme-resource-map.md) | 旧 README 完整原始内容（归档备份） |

### 根目录散文件
| 文件 | 内容 |
|------|------|
| [theory.md](theory.md) | 机器人学理论、RL 基础、控制理论课程 |
| [papers.md](papers.md) | 论文来源（GitHub + arXiv） |
| [motion.md](motion.md) | 动捕数据集与运动生成 |
| [urdf.md](urdf.md) | URDF 模型资源、可视化、开源模型 |
| [retarget.md](retarget.md) | 动作重定向、MoCap、Retarget 相关 |
| [train.md](train.md) | 训练框架（IsaacGym, IsaacLab, RL/IL 框架汇总）|
| [sim2sim.md](sim2sim.md) | 仿真到仿真：Mujoco、PyBullet、Gazebo |
| [sim2real.md](sim2real.md) | 仿真到现实：部署框架、ROS2、经验分享 |

## 使用原则

1. 新资料优先先进入 `sources/`
2. 真正沉淀后的知识，再进入 `wiki/`
3. 不把 `wiki/` 写成纯链接堆
4. 不再让根目录 `README.md` 承担所有资源导航职责

## 与 wiki 的关系

- `sources/` = 输入资料层
- `wiki/` = 结构化知识层

sources 里的内容是原材料，wiki 是提炼后的知识。
