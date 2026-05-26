# 万字长文，读懂人形机器人 AMP：20 篇论文搭起的运动先验圣经

> 来源归档（blog / 微信公众号）

- **标题：** 万字长文，读懂人形机器人 AMP：20 篇论文搭起的运动先验圣经
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w
- **发表日期：** 2026-05-21（frontmatter）
- **入库日期：** 2026-05-21
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 安装的 `wechat-article-for-ai`（Camoufox）；正文约 2.1 万字 / 19 图；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **关联姊妹篇：** [两万字长文，读懂人形机器人强化学习运动控制：42 篇论文搭起的算法圣经](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（`https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA`）
- **一句话说明：** 沿「从能跑到像有身体经验地跑、走、恢复、交互」组织 19 篇 AMP / 运动先验相关论文；核心主张：AMP 不是替代 mimic 的「好看分」，而是把**人类运动分布**嵌入 RL，补足「任务完成之后仍不像一个身体」的缺口；未来形态是**可复用、条件化、可被 VLA / 世界模型调用的身体先验模块**，而非每任务重训的全局判别器。

## 核心摘录（归纳，非全文）

### 问题重框

- **不只问能不能跑**：mimic / 细粒度 reward 都能让机器人跑；AMP 关心 **跑起来之后是否仍像真实身体在运动**。
- **与 mimic 的分工**：mimic 偏逐帧跟轨迹；AMP 偏 **状态转移落在人类运动分布内**，任务可变时仍保持身体合理性。
- **与身体系统栈的关系**：姊妹篇 [42 篇 RL 运动控制综述](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) 搭「八层栈」；本篇补 **控制层里的运动先验 / 风格约束** 这一横切面。

### 五段叙事（原文章节）

| 段 | 主题 | 论文编号 |
|---|---|---|
| **01 分布约束与先验组件化** | AMP 起点、ADD 减 reward 碎片、SMP 可复用先验、Kimodo 动作数据、MotionBricks 实时身体 API | 01–05 |
| **02 人形走跑** | GMP 细轨迹指导、ALMI 上下半身、MoRE 多专家步态、Hiking in the Wild 感知跑酷、统一走跑恢复 | 06–10 |
| **03 多技能与自适应** | Adaptive Humanoid Control、HAML 单策略多技能 AMP | 11–12 |
| **04 交互与长时程** | Humanoid Goalkeeper、HUSKY 滑板、PhysHSI 坐躺站、CLOT 遥操作兜底、TeamHOI masked AMP、Deep Parkour、Embrace Collisions | 13–19 |
| **05 收束** | AMP 思想拆进动作生成 + 运动先验 + 统一控制 + VLA / 世界模型上层 | — |

### 19 篇论文索引（标题以抓取版为准）

| # | 标题 | 机构/备注 |
|---|---|---|
| 01 | AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control | UC Berkeley、上交；SIGGRAPH 2021 线 |
| 02 | Physics-Based Motion Imitation with Adversarial Differential Discriminators (ADD) | SFU、Sony、NVIDIA |
| 03 | SMP: Reusable Score-Matching Motion Priors | SFU、Stanford、NVIDIA 等 |
| 04 | Kimodo: Scaling Controllable Human Motion Generation | NVIDIA；动作数据来源 |
| 05 | MotionBricks: Scalable Real-Time Motions with Smart Primitives | NVIDIA、ETH、UT Austin；G1 实机 |
| 06 | Natural Humanoid Robot Locomotion with Generative Motion Prior (GMP) | 浙江大学 |
| 07 | Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning (ALMI) | 电信 AI、上科大等 |
| 08 | MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits on Complex Terrains | — |
| 09 | Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids | — |
| 10 | Unified Walking, Running, and Recovery via State-Dependent Adversarial Motion Priors | — |
| 11 | Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning | — |
| 12 | HAML: Humanoid Adversarial Multi-Skill Learning via a Single Policy | — |
| 13 | Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints | — |
| 14 | HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control | — |
| 15 | PhysHSI: Towards Real-World Generalizable Humanoid-Scene Interaction | 上海 AI Lab、港科大 |
| 16 | CLOT: Closed-Loop Global Motion Tracking for Whole-Body Humanoid Teleoperation | 上交、上海 AI Lab；arXiv:2602.15060 |
| 17 | TeamHOI: Cooperative Human-Object Interactions with Any Team Size | Garena、Sea AI Lab、NUS |
| 18 | Deep Whole-body Parkour | 清华交叉信息、上海期智 |
| 19 | Embrace Collisions: Humanoid Shadowing for Deployable Contact-Agnostics Motions | 清华交叉信息、上海期智 |

### 作者收束判断（归纳）

1. **AMP = 把人类运动分布嵌入 RL**，不只服务走跑，也服务恢复与长时程稳定性。
2. **算法形态会演化**：ADD / SMP / MotionBricks / masked AMP 指向 **可复用、局部条件化** 先验，而非每任务全局判别器。
3. **不能单独成全部系统**：跑酷、全身接触等需视觉 + 接触 + 时机 + 先验 **多模块合成**（与 [身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) 一致）。
4. **上层 AGI / VLA 会犯错**，底层需要持续 **身体分布边界**（CLOT 式兜底）。

## 对 wiki 的映射

- [humanoid-amp-motion-prior-survey](../../wiki/overview/humanoid-amp-motion-prior-survey.md)（本次升格主页面）
- [amp-reward](../../wiki/methods/amp-reward.md)、[add](../../wiki/methods/add.md)、[smp](../../wiki/methods/smp.md)、[motionbricks](../../wiki/methods/motionbricks.md)、[kimodo](../../wiki/entities/kimodo.md)、[amp-mjlab](../../wiki/entities/amp-mjlab.md)
- [humanoid-rl-motion-control-body-system-stack](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[project-instinct](../../wiki/entities/project-instinct.md)（Deep Parkour / Embrace Collisions）

## 可信度与使用边界

- 标题写「20 篇」、正文编号至 **论文 19**（与 2026-05-21 抓取版一致）；本归档按 **19 篇** 建表，不补第 20 篇臆测条目。
- 各论文方法细节应以官方 PDF / 项目页为准；公众号为 **策展导航**，非唯一一手来源。
- 文中外部图片链接为微信 CDN，wiki 不复述正文。

## 当前提炼状态

- [x] Agent Reach + Camoufox 正文抓取与归纳摘要
- [x] 19 篇论文索引与五段叙事
- [x] wiki 主页面映射确认
- [x] 19 篇论文各建 `sources/papers/humanoid_amp_survey_*` + `wiki/entities/paper-amp-survey-*`（见 [humanoid_amp_survey_19_catalog.md](../papers/humanoid_amp_survey_19_catalog.md)）
