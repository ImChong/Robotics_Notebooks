# CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion

> 来源归档（ingest）

- **标题：** CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion
- **类型：** paper / legged locomotion / teacher-student / PPO / privileged training
- **出处：** arXiv preprint，2024-05-17
- **论文链接：** <https://arxiv.org/abs/2405.10830>
- **PDF：** <https://arxiv.org/pdf/2405.10830>
- **作者：** Hongxi Wang、Haoxiang Luo、Wei Zhang、Hua Chen
- **项目页：** <https://clearlab-sustech.github.io/concurrentTS>
- **相关实现（本 ingest）：** [Robot-Nav/legbot_lab](https://github.com/Robot-Nav/legbot_lab)（`PPO-CTS-MOE` 分支扩展 MoE）
- **入库日期：** 2026-09-16
- **一句话说明：** 提出 **并发 Teacher–Student（CTS）** 架构：教师（特权观测）与学生（可部署观测）在 **同一 PPO 训练循环** 中并行优化，而非先训教师再蒸馏；仿真对比显示相对两阶段 teacher–student 平均速度跟踪误差最多降低约 20%，并在四足与点足双足室内外实验验证。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 论文 | [arXiv:2405.10830](https://arxiv.org/abs/2405.10830) | CTS 原文 |
| 项目页 | [clearlab-sustech.github.io/concurrentTS](https://clearlab-sustech.github.io/concurrentTS) | 视频与补充材料 |
| 代码（Legbot 扩展） | [Robot-Nav/legbot_lab](https://github.com/Robot-Nav/legbot_lab) | Isaac Lab 实现；`PPO` 基线 + `PPO-CTS-MOE` MoE-CTS |
| 相邻范式 | [teacher-student-dagger-training](../../wiki/methods/teacher-student-dagger-training.md) | 两阶段蒸馏 / DAgger 通用范式 |
| 同团队四足 RL | [legbot-mpc-wbc](../../wiki/entities/legbot-mpc-wbc.md) | Robot-Nav 另一开源线（MPC–WBC） |

## 摘要级要点

- **问题：** 传统 teacher–student 先 RL 训特权教师、再监督蒸馏学生；两阶段解耦导致样本效率与最终盲走性能受限。
- **CTS 核心：** 教师与学生 **同时** 与环境交互、共享修改版 PPO 更新；学生从教师 latent 表征学习，教师亦受学生反馈影响。
- **观测不对称：** 教师可用 critic 特权（线速度、地形高度、接触力等）；学生仅本体 + 命令 + 历史。
- **效果：** 相对 SOTA 两阶段 teacher–student，盲 locomotion 平均速度跟踪误差最多 **−20%**；四足与点足双足室内外实验展示鲁棒敏捷运动。
- **Legbot Lab 扩展：** `PPO-CTS-MOE` 在学生编码器引入 **8 专家 MoE + gating**，并加 latent 蒸馏与 load-balance 损失；`PPO` 分支为纯 PPO + 非对称 Actor–Critic + 10 帧历史。

## 核心摘录（面向 wiki 编译）

### 1) 并发训练 vs 两阶段蒸馏

| 维度 | 两阶段 Teacher–Student | CTS |
|------|------------------------|-----|
| 训练顺序 | 先教师 RL，再学生 BC/蒸馏 | 教师与学生 **并行** PPO |
| 样本利用 | 两阶段数据不共享优化 | 同一 rollout 批次联合更新 |
| 表征传递 | 事后蒸馏 | 训练中 latent 对齐 |
| Legbot 读法 | — | `PPO-CTS-MOE` 75% teacher env / 25% student env |

**对 wiki 的映射：** [`wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md`](../../wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md)

### 2) 与 MoE / RoboGauge 线的关系（边界）

- arXiv:2405.10830 **原文为 CTS 架构**，不含 MoE 与 RoboGauge 评测套件。
- [legbot_lab `PPO-CTS-MOE`](https://github.com/Robot-Nav/legbot_lab/tree/PPO-CTS-MOE) 在 CTS 上叠加 **MoE 学生编码器**（工程扩展，README 引用 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) 与 CTS 论文）。
- [RoboGauge](https://robogauge.github.io/)（XJTU，MoE + sim2sim 评测）为 **相邻研究线**，非本文官方代码；选型时勿与 CTS 原文混为一谈。

**对 wiki 的映射：** [`wiki/entities/legbot-lab.md`](../../wiki/entities/legbot-lab.md) 工程实践节

### 3) 开源状态（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| CTS 论文官方代码 | **项目页有演示** | clearlab-sustech 项目页；具体 GitHub 以项目页链接为准 |
| Legbot Lab 实现 | **已开源** | [Robot-Nav/legbot_lab](https://github.com/Robot-Nav/legbot_lab)，Apache-2.0；默认 `PPO`，扩展 `PPO-CTS-MOE` / `WF-CTS-MOE` |
| 权重 / 部署 | **仓库含 ONNX + C++ 部署栈** | `play.py` 导出；`deploy/` + CycloneDDS + 串口网关 |

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md`](../../wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md)**
- 工程实现：**[`wiki/entities/legbot-lab.md`](../../wiki/entities/legbot-lab.md)**
- 交叉：**[`wiki/methods/teacher-student-dagger-training.md`](../../wiki/methods/teacher-student-dagger-training.md)**、**[`wiki/methods/ppo.md`](../../wiki/methods/ppo.md)**、**[`wiki/concepts/privileged-training.md`](../../wiki/concepts/privileged-training.md)**
