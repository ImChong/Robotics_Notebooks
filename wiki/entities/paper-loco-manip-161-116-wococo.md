---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, loco-manip-contact-survey, sequential-contact, reinforcement-learning, whole-body-control, contact-rich, humanoid, cmu, eth]
status: complete
updated: 2026-07-22
arxiv: "2406.06005"
venue: "CoRL 2024 Oral"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-05-mocap-human-video.md
  - ../overview/loco-manip-contact-category-04-post-contact-stability.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/papers/loco_manip_161_survey_116_wococo.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
summary: "WoCoCo（CoRL 2024 Oral, arXiv:2406.06005）研究顺序接触下的全身控制，展示 clap dance、parkour jumping、bidirectional cliff traversal、anywhere-to-anywhere loco-manip 与多端接触 dino ball manipulation；官方 LeCAR-Lab/wococo 开源 legged_gym/RSL-RL 示例。"
---

# WoCoCo

**WoCoCo**（*Learning Whole-Body Humanoid Control with Sequential Contacts*）把人形控制的难点从单次接触扩展到 **顺序接触**：脚、手、身体不同部位在任务中按阶段建立/断开接触，策略必须在接触切换中维持全身稳定和任务进展。

## 一句话定义

WoCoCo 是一个顺序接触全身控制框架，用 RL 学习跨接触阶段的人形/多体机器人运动与 loco-manip 技能。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WoCoCo | Whole-Body Control with Sequential Contacts | 本文方法名 |
| RL | Reinforcement Learning | 策略学习主框架 |
| MDP | Markov Decision Process | README 鼓励按任务扩展 reward/MDP |
| RSL-RL | Rudin et al. legged RL library | 官方代码依赖的 RL 框架 |
| IsaacGym | NVIDIA Isaac Gym | 官方代码仿真后端 |
| H1 | Unitree H1 Humanoid | README 示例任务 `h1:jumpjack` |

## 为什么重要

- **接触有时序，不是静态标签**：clap、jump、cliff traversal、dribble ball 都要求多个身体部位按阶段接触。
- **与 OmniContact 的接触流问题意识相邻**：两者都强调接触建立/保持/切换/断开，只是 WoCoCo 更偏底层 RL 控制。
- **展示跨形态迁移思想**：项目页把 dino 用左前脚、右前脚、后脚、头、尾等不同端点控球，说明 sequential contact 不限于人形手脚。
- **官方代码可跑但偏示例化**：README 明确提供 clap-and-dance 示例，并鼓励用户为具体任务工程化 reward/MDP。

## 流程总览

```mermaid
flowchart TB
  task["sequential-contact task"] --> phases["接触阶段设计\nclap/jump/traverse/dribble"]
  phases --> mdp["MDP + reward implementation"]
  mdp --> train["legged_gym + RSL-RL training"]
  train --> policy["whole-body policy"]
  policy --> sim["humanoid / dino rollouts"]
  sim --> contact["多端接触切换与稳定"]
```

## 核心原理（详细）

WoCoCo 的核心价值在于将全身运动任务写成顺序接触问题：不同阶段允许/鼓励不同身体部位承担支撑、推动、击打或操控。相较纯 motion tracking，它更关注 contact schedule 与 reward 如何让策略主动建立、利用和退出接触。

官方 README 也很诚实：代码「de-engineered」了很多任务特定技巧，提供 clap-and-dance 的 reward 与 MDP 示例，鼓励研究者根据地形设计、curiosity observation、sim-to-real 技巧自行工程化。这说明 WoCoCo 更像一个顺序接触 RL 模板，而不是开箱即用的通用人形操作系统。

## 评测与结果

WoCoCo（CoRL 2024 Oral）的评测以**顺序接触任务的定性演示**为主，展示策略能在多阶段接触切换中维持全身稳定与任务进展：

- **人形任务**：clap dance（拍手舞）、parkour jumping、bidirectional cliff traversal、anywhere-to-anywhere loco-manip 等，均要求脚/手/身体不同部位按阶段建立与断开接触。
- **跨形态演示**：dino 用左前脚、右前脚、后脚、头、尾等不同端点控球（ball manipulation），说明 sequential contact 框架不限于人形手脚，可迁移到多体机器人。
- **能力边界**：官方 README 明确代码「de-engineered」掉许多任务特定技巧，仅提供 clap-and-dance 的 reward/MDP 示例；因此评测更多体现**框架能表达顺序接触技能**，而非开箱即用的统一成功率基准。

> 指标说明：归档材料以项目页定性演示为主，未见跨任务统一的量化成功率表；本页对具体数字只做索引级记录，以论文与项目页为准。

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    actor U as 用户
    participant Repo as LeCAR-Lab/wococo
    participant Gym as IsaacGym + legged_gym
    participant Train as scripts/train.py
    participant Play as scripts/play.py
    U->>Repo: git clone + pip install -r requirements.txt
    U->>Gym: 安装 IsaacGym Preview4
    U->>Train: python3 scripts/train.py --task=h1:jumpjack
    Train->>Gym: RSL-RL 训练顺序接触策略
    Gym-->>Train: policy checkpoint
    U->>Play: python scripts/play.py --task=h1:jumpjack --num_envs=3
    Play-->>U: clap/jump 等 rollout 可视化
```

## 工程实践（含开源状态）

| 项 | 结论 |
|----|------|
| 项目页 | <https://lecar-lab.github.io/wococo/> |
| 官方代码 | <https://github.com/LeCAR-Lab/wococo>，CC BY-NC 4.0 + inherited licenses |
| 依赖 | Python 3.8、IsaacGym Preview4、legged_gym、RSL-RL |
| 入口 | `legged_gym/legged_gym/scripts/train.py --task=h1:jumpjack` |
| 注意 | 非商业许可；示例框架需按任务工程化 |

## 与其他工作对比

WoCoCo 关注身体层的顺序接触控制，与 [OmniContact](./paper-omnicontact-humanoid-loco-manipulation.md)、[FALCON](./paper-loco-manip-161-109-falcon.md) 在「接触如何被表示与组织」上相邻但取舍不同。下表为定性对照。

| 维度 | WoCoCo | OmniContact | FALCON |
|------|--------|-------------|--------|
| 核心问题 | 顺序接触全身控制（多部位分阶段接触） | 可组合 meta-skill 的接触流接口 | 受力 loco-manip 的稳定与力补偿 |
| 接触抽象 | 任务级 contact schedule + 手写 reward/MDP | Contact Flow（稀疏体目标 + 四端接触时序） | 3D force curriculum，无显式接触时序表示 |
| 组合/恢复 | 示例化，需按任务工程化 | CF-Gen 链式组合 + 50 Hz 在线重规划 | 跨 embodiment 迁移，无链式组合叙事 |
| 支持形态 | 人形 + dino 等多体 | Unitree G1 | Unitree G1 / Booster T1 |
| 评测形态 | 定性演示（clap/jump/cliff/dribble） | 定量仿真成功率 + 自采数据集 | 真机力范围 + 2× 跟踪精度 |
| 开源 | CC BY-NC，示例框架 | sim2sim + 数据集开放 | MIT 全链路 |

## 局限与风险

- **不是完整产品级复现**：README 明确鼓励用户自己工程化环境和 sim-to-real。
- **任务 reward 仍需设计**：顺序接触能力依赖 MDP/reward 组织。
- **商业使用受限**：CC BY-NC 4.0 与继承许可证限制。
- **与视觉/语言无直接耦合**：WoCoCo 解决身体层顺序接触，不处理高层语义调度。

## 关联页面

- [Loco-Manip 接触分类 04：接触后如何稳住](../overview/loco-manip-contact-category-04-post-contact-stability.md)
- [161 篇 · 05 动捕、人类视频与交互动作规划](../overview/loco-manip-161-category-05-mocap-human-video.md)
- [OmniContact](./paper-omnicontact-humanoid-loco-manipulation.md)
- [FALCON](./paper-loco-manip-161-109-falcon.md)
- [Whole-Body Control](../concepts/whole-body-control.md)

## 参考来源

- [loco_manip_161_survey_116_wococo.md](../../sources/papers/loco_manip_161_survey_116_wococo.md)
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- [loco-manip-contact-category-04-post-contact-stability](../overview/loco-manip-contact-category-04-post-contact-stability.md)
- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
- Zhang et al., *WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts*, arXiv:2406.06005 / CoRL 2024 Oral. <https://arxiv.org/abs/2406.06005>
- 官方代码：<https://github.com/LeCAR-Lab/wococo>

## 推荐继续阅读

- [WoCoCo 项目页](https://lecar-lab.github.io/wococo/)
- [WoCoCo GitHub](https://github.com/LeCAR-Lab/wococo)
- [OmniContact](./paper-omnicontact-humanoid-loco-manipulation.md)
