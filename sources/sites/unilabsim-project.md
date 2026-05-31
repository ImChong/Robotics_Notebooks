# UniLab 项目页（unilabsim.github.io）

> 来源归档

- **标题：** UniLab — A Heterogeneous Training Framework for Embodied Reinforcement Learning
- **类型：** site
- **URL：** <https://unilabsim.github.io>
- **论文：** <https://arxiv.org/abs/2605.30313>
- **代码：** <https://github.com/unilabsim/UniLab>
- **入库日期：** 2026-05-31
- **一句话说明：** 产品叙事与演示：3–10× 端到端加速、macOS 一等公民、双物理后端、多平台/多算法/多任务族、与 GPU-centric 栈对比表、浏览器 MotrixSim 策略试玩、to-real 视频。
- **沉淀到 wiki：** [UniLab](../../wiki/entities/unilab.md)

---

## 页面能力要点（策展）

1. **CPU sim, GPU learn**：独立 CPU rollout worker → 无锁共享内存 → GPU learner；异步权重同步缓解 tightly-coupled 流水线互等。
2. **macOS 一等目标**：Apple Silicon（MPS/MLX）端到端训练，同代码路径覆盖 CUDA / ROCm / Intel XPU。
3. **算法覆盖**：on-policy（PPO、APPO、HIM-PPO）、off-policy（SAC、TD3、FastSAC、FlashSAC）、蒸馏（HORA）。
4. **对比表**：相对 IsaacLab、IsaacGym、mjlab、Genesis、IsaacSim — UniLab 标 **异构 runtime 全支持**、非 GPU-resident sim。
5. **结果叙事**：G1 Flip 3.3×、G1 Walk Flat 8.4×、G1 Motion Tracking 11.0× 等代表任务墙钟（与论文一致口径）。
6. **To-real**：六类真机任务概览视频；任务卡片链 MotrixSim 浏览器 demo。
7. **跨平台表**：M5 Max / RTX4090+9950X3D / AMD ROCm / Intel Arc 等代表配置与训练任务。

## BibTeX / 关联后端论文（页面 Cite 区）

- UniLab：arXiv:2605.30313
- MuJoCoUni：arXiv:2605.24922
- MotrixSim：软件引用（Motphys Team）

## 对 wiki 的映射

- [UniLab](../../wiki/entities/unilab.md) — 演示、对比与工程叙事
- [sources/papers/unilab_arxiv_2605_30313.md](../papers/unilab_arxiv_2605_30313.md) — 方法与实验以论文为准
