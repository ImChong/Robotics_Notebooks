# 《Science Robotics》同期三篇人形机器人论文：从视频动作到运动基础模型，再到视觉踢球

> 来源归档（blog / 微信公众号）

- **标题：** 《Science Robotics》同期三篇人形机器人论文：从视频动作到运动基础模型，再到视觉踢球：人形机器人开始进入下一阶段
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/UC-LTs_E83ssuImnXusQGA
- **发表日期：** 2026-08-27
- **入库日期：** 2026-08-27
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始抓取落盘：** [wechat_embodied_ai_lab_scirobotics_three_humanoid_papers_2026-08-27.md](../raw/wechat_embodied_ai_lab_scirobotics_three_humanoid_papers_2026-08-27.md)
- **姊妹长文：** [42 篇 RL 运动控制](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（身体系统栈总框架）
- **一句话说明：** 把 *Science Robotics* 11(117) 同期三篇——[ZEST](../../wiki/entities/paper-zest.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[视觉驱动反应式足球](../../wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md)——读成**三个系统层级**（技能编译器 / 运动基础模型 / 感知任务策略），而不是互相竞争的通用 tracker。

## 核心摘录（归纳，非全文）

### 总判断

三篇都用 RL + Sim2Real，但回答的不是同一科学问题：

| 层级 | 论文 | 一句话站位 |
|------|------|------------|
| **技能编译器** | ZEST | 参考动作 → 可部署专项策略的自动化配方；跨 Atlas / G1 / Spot **方法复用**，不是权重复用 |
| **运动基础模型** | SONIC | 数据 / 模型 / 算力三轴 scaling + 共享动作 token，给 VLA / VR / 视频一个身体接口 |
| **感知任务策略** | 视觉足球 | 把噪声、延迟、漏检写进统一 RL 环，搜球–追球–踢球连续过渡 |

文内强调：**真正的机器人基础模型更可能是分层基础设施，而不是一个巨大端到端网络。**

### 易混概念（文内纠偏）

- **ZEST 的 zero-shot** = 仿真策略不经真机微调直接部署，不是「无需训练就会新技能」。
- **SONIC 的统一** = 一个策略覆盖大规模运动分布 + token 接口；对比数字混合了算法、数据与重定向，不能全归因于网络结构。
- **视觉足球的「视觉驱动」** = 结构化检测结果进策略，**不是** RGB 像素直接出力矩；检测器仍是独立前端。
- **跨本体** 目前仍是「同一配方分别训练」，不是同一组权重在 Atlas / G1 / Spot 之间直接跑。

### 开源核查（步骤 2.5，2026-08-27）

| 工作 | 项目页 | 结论 |
|------|--------|------|
| ZEST | 无项目页 | **确认未开源**（与既有实体页一致） |
| SONIC | [GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/) | **已开源**：页头写 *Science Robotics* 11(117)，DOI `10.1126/scirobotics.aed4592`；代码 [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)，权重 HF `nvidia/GEAR-SONIC`。页上另写 “All models shown in the videos will be released”，以已发布仓/权重为准 |
| 视觉足球 | [humanoid-kick.github.io](https://humanoid-kick.github.io/) | **部分开源**：Zenodo 仿真训练/推理 + checkpoint；无 GitHub；真机栈未发布 |

## 对 wiki 的映射

- [ZEST vs SONIC vs 视觉足球（三层对比）](../../wiki/comparisons/zest-vs-sonic-vs-vision-soccer.md) — 本次升格主页面
- [ZEST 论文实体](../../wiki/entities/paper-zest.md) / [ZEST 方法](../../wiki/methods/zest.md)
- [SONIC](../../wiki/methods/sonic-motion-tracking.md)
- [Vision-Driven Reactive Soccer](../../wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md)
- [人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)
- [Humanoid Soccer](../../wiki/tasks/humanoid-soccer.md)

## 可信度与使用边界

- 第三方技术解读，不是论文原文；数字与开源状态以 DOI / 项目页 / 既有 wiki 实体为准。
- 文内「Deep Whole-body Parkour / Hiking in the Wild 投稿中」等时间线条目随发表状态变化，不在本条升格为新实体。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 三篇均复用既有 wiki 节点，不重复造页
- [x] SONIC 正式刊物信息按 GEAR-SONIC 项目页补进方法页
- [x] 升格三层对比页
