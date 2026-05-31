# UniLab（官方仓库）

> 来源归档

- **标题：** UniLab — Heterogeneous Training Framework for Embodied RL
- **类型：** repo
- **代码：** <https://github.com/unilabsim/UniLab>
- **论文：** <https://arxiv.org/abs/2605.30313>
- **项目页：** <https://unilabsim.github.io>
- **入库日期：** 2026-05-31
- **一句话说明：** 完整可扩展训练系统：CPU 批量物理（MuJoCoUni / MotrixSim）+ GPU learner + 共享内存 IPC；统一训练/评测入口与 task/backend 接口。
- **沉淀到 wiki：** [UniLab](../../wiki/entities/unilab.md)

---

## 仓库要点（ingest 快照）

| 维度 | 内容 |
|------|------|
| **定位** | 异构 **CPU-sim / GPU-learn** 机器人 RL 训练栈，非「又一个 GPU 驻留仿真器」 |
| **物理后端** | MuJoCoUni、MotrixSim（CPU batch） |
| **算法** | PPO、APPO、SAC、TD3、FlashSAC 等（以 README / 文档为准） |
| **平台** | Linux CUDA、Apple Silicon（MPS/MLX）、AMD ROCm、Intel XPU |
| **机器人** | 四足（Go1/Go2）、人形（G1）、灵巧手（Allegro/Sharpa）、轮足（Go2w）等 |
| **IPC** | 主机共享内存缓冲 + 无锁权重发布；减少采集器与 learner 互等 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) | GPU 驻留、强耦合基线对照 |
| [mjlab](../../wiki/entities/mjlab.md) | 同为 MuJoCo 生态 GPU 路径；论文含 MjLab/MJWarp 对比 |
| [Motrix](../../wiki/entities/motrix.md) | UniLab 可选 CPU 物理后端之一 |
| [MuJoCo](../../wiki/entities/mujoco.md) | MuJoCoUni 批量 runtime 建立在 MuJoCo 语义之上 |
| [仿真器选型](../../wiki/queries/simulator-selection-guide.md) | 补充「异构 runtime」选型维度 |

## 对 wiki 的映射

- 主实体页：**`wiki/entities/unilab.md`**
- 论文摘录：**`sources/papers/unilab_arxiv_2605_30313.md`**
- 项目页：**`sources/sites/unilabsim-project.md`**
