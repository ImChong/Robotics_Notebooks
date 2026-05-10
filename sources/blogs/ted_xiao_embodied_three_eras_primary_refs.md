# 具身智能「三个时代」叙事：技术话题与一手文献索引

> **收录类型**：`blogs` + 一手文献指针（论文 / 官方技术报告 / 机构博客）  
> **触发编译**：机器之心对 RoboPapers 访谈的二次整理（见下方「二手编译入口」），本文仅抽取其中出现的**可核验技术实体**，并为每条给出可追溯的一手出处。

---

## 二手编译入口（非一手事实来源）

| 资料 | 链接 | 说明 |
|------|------|------|
| 微信公众号编译稿 | https://mp.weixin.qq.com/s/YJYy7dRGUbykxng2gEt9gw | 中文综述；其中的组织过程叙事（例如内部代号、时间线）应以论文与官方发布为准 |
| RoboPapers 访谈视频 | https://www.youtube.com/watch?v=etPqBphTgmE | Ted Xiao 口述来源 |

---

## 时代 A：端到端 RL / 大规模模仿学习与「存在性证明」

### 深度强化学习与游戏基准（叙事背景）

| 话题 | 一手资料 |
|------|-----------|
| DQN / Atari | Mnih et al., *Playing Atari with Deep Reinforcement Learning*, arXiv:1312.5602 — https://arxiv.org/abs/1312.5602 |
| AlphaGo | Silver et al., *Mastering the game of Go with deep neural networks and tree search*, Nature (2016)；摘要页 https://www.deepmind.com/research/alphago-zero-learning-from-scratch |

### 真实机器人视觉操控 RL：QT-Opt 与系统栈

| 话题 | 一手资料 |
|------|-----------|
| QT-Opt（连续动作 Q 学习 + 大规模真实抓取） | Kalashnikov et al., *Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation*, arXiv:1806.10293 — https://arxiv.org/abs/1806.10293 |
| 配套博文（系统规模、数据闭环） | Google Research Blog, *Scalable Deep Reinforcement Learning for Robotic Manipulation* — https://blog.research.google/2018/06/scalable-deep-reinforcement-learning.html |

### 大规模语言条件模仿与多任务 RL

| 话题 | 一手资料 |
|------|-----------|
| BC-Z（语言条件模仿 / teleop 数据规模化） | Jang et al., *BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning*, arXiv:2202.02005 — https://arxiv.org/abs/2202.02005 |
| MT-Opt（连续多任务机器人 RL） | Gupta et al., *MT-Opt: Continuous Multi-Task Robotic Reinforcement Learning at Scale*, arXiv:2104.08212 — https://arxiv.org/abs/2104.08212 |
| Google AI Blog（MT-Opt 数据规模与任务定义） | https://ai.googleblog.com/2021/04/multi-task-robotic-reinforcement.html |

### 从「玩耍」数据中学习（与事后重标记思想相邻）

| 话题 | 一手资料 |
|------|-----------|
| Learning from Play / Play-LMP | Lynch et al., *Learning Latent Plans from Play*, CoRL 2019 / arXiv:1903.01973 — https://arxiv.org/abs/1903.01973 ，项目页 https://learning-from-play.github.io/ |

### 目标条件 RL 中的事后经验（HER）

| 话题 | 一手资料 |
|------|-----------|
| Hindsight Experience Replay | Andrychowicz et al., *Hindsight Experience Replay*, arXiv:1707.01495 — https://arxiv.org/abs/1707.01495 |

### 仿真 — 真实图像风格迁移（叙事中与域偏移对策一并提及）

| 话题 | 一手资料 |
|------|-----------|
| CycleGAN | Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, arXiv:1703.10593 — https://arxiv.org/abs/1703.10593 |

---

## 时代 B：语言模型 × 机器人 —— 规划与「基础策略」形态

### SayCan：语言模型规划 + 价值可行性

| 话题 | 一手资料 |
|------|-----------|
| SayCan（Do As I Can） | Ahn et al., *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*, arXiv:2204.01691 — https://arxiv.org/abs/2204.01691 |

### RT-1 / RT-2：Transformer 策略与 VLA

| 话题 | 一手资料 |
|------|-----------|
| RT-1 | Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*, arXiv:2212.06817 — https://arxiv.org/abs/2212.06817 |
| RT-1 官方博文 | https://research.google/blog/rt-1-robotics-transformer-for-real-world-control-at-scale/ |
| RT-2（VLA：视觉 — 语言 — 动作） | Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, arXiv:2307.15818 — https://arxiv.org/abs/2307.15818 |

### DIAL：用 VLM 对离线轨迹做指令增强（「自动打语言标签」）

| 话题 | 一手资料 |
|------|-----------|
| DIAL（Data-driven Instruction Augmentation for Language-conditioned control） | Xiao et al., *Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models*, arXiv:2211.11736 — https://arxiv.org/abs/2211.11736 ，项目页 https://instructionaugmentation.github.io/ |

### Open X-Embodiment：跨本体规模化数据

| 话题 | 一手资料 |
|------|-----------|
| Open X-Embodiment | Padalkar et al., *Open X-Embodiment: Robotic Learning at Scale*, arXiv:2310.08864 — https://arxiv.org/abs/2310.08864 |

---

## 时代 C：规模化竞赛 —— VLA、评测、数据形态与公司发布

### VLA 开源 / 通用策略代表（叙事中与 RT 线并列讨论）

| 话题 | 一手资料 |
|------|-----------|
| Octo（开源 generalist policy） | Ghosh et al., *Octo: An Open-Source Generalist Robot Policy*, arXiv:2405.12213 — https://arxiv.org/abs/2405.12213 |
| π₀（Physical Intelligence，叙事中 π 系列代际） | Black et al., *π₀: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164 — https://arxiv.org/abs/2410.24164 |

### 高频遥操作硬件与双臂数据（叙事：解锁灵巧操作）

| 话题 | 一手资料 |
|------|-----------|
| ALOHA / Mobile ALOHA | Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, arXiv:2304.13705 — https://arxiv.org/abs/2304.13705 ；*Mobile ALOHA* arXiv:2401.02117 — https://arxiv.org/abs/2401.02117 |

### Google DeepMind：Gemini Robotics 产品线（官方发布与技术报告）

| 话题 | 一手资料 |
|------|-----------|
| Gemini Robotics & Gemini Robotics-ER（首次集中发布） | Google DeepMind Blog — https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/ |
| Gemini Robotics 1.5 / ER 1.5 | Google DeepMind Blog — https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/ |
| 技术报告 PDF（长文细节） | https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-1.5-Tech-Report.pdf |

### 分布式真实世界评测

| 话题 | 一手资料 |
|------|-----------|
| RoboArena | Atreya et al., *RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies*, arXiv:2506.18123 — https://arxiv.org/abs/2506.18123 ，代码 https://github.com/robo-arena/roboarena |

### 大规模人类 / 交互数据预训练（叙事中的产业侧）

| 话题 | 一手资料 |
|------|-----------|
| Generalist AI（GEN 系列与数据规模主张） | 机构博客入口 https://generalistai.com/blog/nov-04-2025-GEN-0 与 https://generalistai.com/blog/apr-02-2026-GEN-1 （具体数字随发布迭代，引用时以页面为准） |

### 世界模型与策略（叙事中「World Models / Video Action Models」方向）

| 话题 | 一手资料 |
|------|-----------|
| 仓库内综述索引 | [rl_foundation_models.md](../papers/rl_foundation_models.md)（含 TD-MPC2 等 world-model 路线指针） |

---

## 与 wiki 的映射建议

| 知识库页面 | 对应技术线 |
|------------|------------|
| [wiki/queries/robot-learning-three-eras-narrative.md](../../wiki/queries/robot-learning-three-eras-narrative.md) | 叙事透镜 + 全索引入口 |
| [wiki/concepts/deep-rl-game-milestones.md](../../wiki/concepts/deep-rl-game-milestones.md) | DQN / AlphaGo |
| [wiki/methods/qt-opt.md](../../wiki/methods/qt-opt.md) | QT-Opt |
| [wiki/methods/bc-z.md](../../wiki/methods/bc-z.md) | BC-Z |
| [wiki/methods/mt-opt.md](../../wiki/methods/mt-opt.md) | MT-Opt |
| [wiki/methods/learning-from-play-lmp.md](../../wiki/methods/learning-from-play-lmp.md) | Learning from Play |
| [wiki/methods/her.md](../../wiki/methods/her.md) | HER |
| [wiki/methods/cyclegan-sim2real.md](../../wiki/methods/cyclegan-sim2real.md) | CycleGAN |
| [wiki/methods/saycan.md](../../wiki/methods/saycan.md) | SayCan |
| [wiki/methods/robotics-transformer-rt-series.md](../../wiki/methods/robotics-transformer-rt-series.md) | RT-1 / RT-2 |
| [wiki/methods/dial-instruction-augmentation.md](../../wiki/methods/dial-instruction-augmentation.md) | DIAL |
| [wiki/concepts/open-x-embodiment.md](../../wiki/concepts/open-x-embodiment.md) | Open X-Embodiment |
| [wiki/methods/octo-model.md](../../wiki/methods/octo-model.md) | Octo |
| [wiki/methods/π0-policy.md](../../wiki/methods/π0-policy.md) | π₀ |
| [wiki/entities/aloha.md](../../wiki/entities/aloha.md) | ALOHA |
| [wiki/entities/gemini-robotics.md](../../wiki/entities/gemini-robotics.md) | Gemini Robotics |
| [wiki/methods/roboarena.md](../../wiki/methods/roboarena.md) | RoboArena |
| [wiki/entities/generalist-ai-robotics.md](../../wiki/entities/generalist-ai-robotics.md) | Generalist AI |
| [wiki/methods/generative-world-models.md](../../wiki/methods/generative-world-models.md) | World Models / 视频模型叙事 |
| [wiki/concepts/foundation-policy.md](../../wiki/concepts/foundation-policy.md) | Foundation Policy 总览 |
| [wiki/methods/vla.md](../../wiki/methods/vla.md) | VLA |
| [wiki/methods/imitation-learning.md](../../wiki/methods/imitation-learning.md) | 模仿学习总览 |
| [wiki/methods/reinforcement-learning.md](../../wiki/methods/reinforcement-learning.md) | 强化学习总览 |

---

## 参考来源

- 一手条目：上表各 arXiv / 官方博客 / 技术报告 PDF  
- 触发编译：微信公众号 https://mp.weixin.qq.com/s/YJYy7dRGUbykxng2gEt9gw  
