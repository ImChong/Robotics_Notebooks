# UniLab 项目页（unilabsim.github.io）

> 来源归档

- **标题：** UniLab — A Heterogeneous Training Framework for Embodied Reinforcement Learning
- **类型：** site
- **URL：** <https://unilabsim.github.io>
- **论文：** <https://arxiv.org/abs/2605.30313>
- **代码：** <https://github.com/unilabsim/UniLab>
- **入库日期：** 2026-05-31（初入库）；**复核：** 2026-09-18
- **会议：** CoRL 2026 **Accepted**（项目页 banner，2026-09 复核）
- **一句话说明：** 产品叙事与演示：3–10× 端到端加速、macOS 一等公民、双物理后端、**14 个已上线任务**、7 类算法、5 类机器人、与 GPU-centric 栈对比表、**17 项**浏览器 MotrixSim 策略试玩、六类 to-real 视频、跨平台墙钟训练表。
- **沉淀到 wiki：** [UniLab](../../wiki/entities/unilab.md)

---

## 页面能力要点（策展）

1. **CPU sim, GPU learn**：独立 CPU rollout worker → 无锁共享内存 → GPU learner；异步权重同步缓解 tightly-coupled 流水线互等。
2. **macOS 一等目标**：Apple Silicon（MPS/MLX）端到端训练，同代码路径覆盖 CUDA / ROCm / Intel XPU。
3. **算法覆盖**：on-policy（PPO、APPO、HIM-PPO）、off-policy（SAC、TD3、FastSAC、FlashSAC）、蒸馏（HORA）。
4. **对比表**：相对 IsaacLab、IsaacGym、mjlab、Genesis、IsaacSim — UniLab 标 **异构 runtime 全支持**、非 GPU-resident sim。
5. **结果叙事**：G1 Flip 3.3×、G1 Walk Flat 8.4×、G1 Motion Tracking 11.0× 等代表任务墙钟（与论文一致口径）。
6. **To-real**：六类真机任务概览视频；任务卡片链 MotrixSim 浏览器 demo。
7. **跨平台表**：M5 Max / RTX4090+9950X3D / AMD ROCm / Intel Arc 等代表配置与训练任务；附录给出各 (task, platform) wall-clock 分钟数（如 FastSAC G1 Walk Flat：4090 18.3 min vs M5 Max 18.8 min vs XPU 185H 115.4 min）。
8. **已上线任务规模（页面统计条）：** 2 物理后端 · 4 平台 · **5** 机器人类别（arms / quadrupeds / humanoids / hands / wheeled-leg）· **7** 算法 · **7** 任务族 · **14 tasks shipped**。
9. **浏览器 MotrixSim demo（策展，2026-09）：** Go1/Go2/Go2w joystick & handstand；G1 walk/dance/shuttle/box/flip/climb；Sharpa in-hand；Go2+airbot loco-manip；Stewart 6-DOF ball balancing 等 **17** 卡片，多数标注 PPO/SAC 与训练 terrain。

## 浏览器 demo 任务清单（项目页 Try policies 区）

| 任务卡片 | 机体 | 算法 / 备注 |
|----------|------|-------------|
| Joystick (flat/rough) | Go1 / Go2 | PPO (torch) |
| Handstand | Go2 | PPO |
| Joystick rough tiles | Go2w | PPO |
| Walk flat / rough | G1 | SAC |
| Dance (SAC WBT / motion tracking) | G1 | SAC / PPO |
| Shuttle run / box / backflip / wall back-flip / climb | G1 | PPO / object tracking |
| In-hand reorient | Sharpa | PPO / APPO |
| Loco-manipulation | Go2 + airbot | PPO |
| Ball balancing | 6-DOF Stewart | PPO |

## BibTeX / 关联后端论文（页面 Cite 区）

- UniLab：arXiv:2605.30313
- MuJoCoUni：arXiv:2605.24922
- MotrixSim：软件引用（Motphys Team）

## 对 wiki 的映射

- [UniLab](../../wiki/entities/unilab.md) — 演示、对比与工程叙事
- [sources/papers/unilab_arxiv_2605_30313.md](../papers/unilab_arxiv_2605_30313.md) — 方法与实验以论文为准
