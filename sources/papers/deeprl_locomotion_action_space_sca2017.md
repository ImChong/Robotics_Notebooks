# deeprl_locomotion_action_space_sca2017

> 来源归档（ingest）

- **标题：** Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter?
- **作者：** Xue Bin Peng, Michiel van de Panne
- **类型：** paper
- **venue：** ACM SIGGRAPH / Eurographics Symposium on Computer Animation (SCA 2017) — **Best Student Paper Award**
- **预印本：** [arXiv:1611.01055](https://arxiv.org/abs/1611.01055)（2016-11-03）
- **正式出版：** [DOI 10.1145/3099564.3099567](https://doi.org/10.1145/3099564.3099567)
- **入库日期：** 2026-05-21
- **一句话说明：** 在步态周期模仿任务上系统对比 **扭矩、肌肉激活、目标关节角（PD）、目标关节角速度** 四种动作参数化对深度 RL 学习速度、策略鲁棒性、运动质量与策略查询率的影响，论证 **带局部反馈的高层动作空间** 往往优于端到端扭矩控制。
- **沉淀到 wiki：** 是 → [wiki/entities/paper-deeprl-locomotion-action-space-sca2017.md](../../wiki/entities/paper-deeprl-locomotion-action-space-sca2017.md)

## 为什么值得保留

- **动作接口选型的「前史」**：仓库内 [RL+PD 动作接口论文索引](rl_pd_action_interface_locomotion.md) 与 [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md) 大量讨论「目标角 + PD」与「直驱扭矩」；本文在 **平面角色、模仿步态** 设定下给出 **可复现的四种语义对照**，是 Peng 后续 Cassie / sim2real 路线的理论起点之一。
- **与 Peng 学位论文衔接**：作者 [M.Sc. 论文](https://xbpeng.github.io/projects/MSc_Thesis/index.html) *Developing Locomotion Skills with Deep Reinforcement Learning*（2017）与主页 [ActionSpace 项目页](https://xbpeng.github.io/projects/ActionSpace/index.html) 把同一研究线公开，便于 curator 核对 arXiv 与 SCA 终稿差异。

## 相关资料（原始链接）

| 类型 | 链接 |
|------|------|
| 项目页 + PDF + 视频 | <https://xbpeng.github.io/projects/ActionSpace/index.html> |
| arXiv 摘要 | <https://arxiv.org/abs/1611.01055> |
| arXiv PDF | <https://arxiv.org/pdf/1611.01055.pdf> |
| UBC 作者组 PDF 镜像 | <https://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/2017-TOG-deepLoco.pdf> |
| OpenReview（ICLR 2017 投稿记录） | <https://openreview.net/forum?id=r1kGbydxg> |
| Peng M.Sc. 论文 | <https://xbpeng.github.io/projects/MSc_Thesis/MSc_Thesis.pdf> |

## 核心论文摘录

### 1) 研究问题与 MDP 设定

- **问题：** 深度 RL 可用高维状态描述符，但 **动作表示如何选** 对样本效率、鲁棒性与运动质量的影响当时缺乏系统实证。
- **任务：** 多种 **平面关节角色** 上的 **步态周期模仿（gait-cycle imitation）**，覆盖多种步态；评价维度包括 **学习时间、策略鲁棒性、运动质量、策略查询率（policy query rate）**。
- **对 wiki 的映射：**
  - [DeepRL 动作空间对比（SCA 2017）](../../wiki/entities/paper-deeprl-locomotion-action-space-sca2017.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 2) 四种动作参数化（对照轴）

| 记号 | 语义 | 要点 |
|------|------|------|
| **Tor** | 关节扭矩 | 最低层、探索空间大；学习慢、对扰动敏感 |
| **MTU** | 肌肉–肌腱单元激活 | 生物力学驱动；利用被动动力学与本体反馈 |
| **PD** | 目标关节角 + PD 局部反馈 | 高层目标 + 内环跟踪；样本效率与运动质量通常更优 |
| **Vel** | 目标关节角速度 | 介于扭矩与 PD 之间；依赖速度级反馈 |

- **核心结论（归纳）：** **带局部反馈的高层参数化**（尤其 **PD 目标角**）在多数设定下 **显著缩短学习、提高鲁棒性与运动质量**，同时仍保留相对扭矩级的 **通用性**；被动动力学与 **具身反馈（embodied feedback）** 对有效运动塑造很重要。
- **对 wiki 的映射：**
  - [Legged / Humanoid RL 中 Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md) — 「为何 PD 目标空间」的 **图形学/角色侧** 最早系统证据
  - [Cassie 反馈控制 DRL](../../wiki/entities/paper-cassie-feedback-control-drl.md) — 机器人侧后续把 PD 语义写进 Cassie MDP 的代表作

### 3) 与后续机器人工作的关系（curator 注）

- 本文实验体为 **物理仿真角色（平面）**，不等价于 **真机 Cassie / 四足** 结论；但 **「低维目标 + 已知内环」降低 RL 负担** 的论证与 [1803.05580 Cassie feedback DRL](https://arxiv.org/abs/1803.05580)、[2203.05194 四足扭矩 RL](https://arxiv.org/abs/2203.05194) 形成 **同一条研究线上的前后对照**。
- **对 wiki 的映射：**
  - [Xue Bin Peng](../../wiki/entities/xue-bin-peng.md)
  - [Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md)

## 当前提炼状态

- [x] 一手链接、四种动作语义表与 SCA 2017 获奖信息已归档
- [x] 已升格独立 wiki 实体页并接入 [RL+PD 动作接口论文索引](rl_pd_action_interface_locomotion.md)
- [~] 若后续精读 PDF，可补各角色/步态下的定量排序表（当前以项目页摘要与公开二次解读为准）
