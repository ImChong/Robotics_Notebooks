# Physical Self-Play（Skild AI）

> 来源归档（blog / 公司官方）

- **标题：** Physical Self-Play
- **类型：** blog
- **作者 / 组织：** Skild Team
- **原始链接：** <https://www.skild.ai/blogs/physical-self-play>
- **发表日期：** 2026-09-23
- **入库日期：** 2026-09-24
- **抓取方式：** 官方页直连（WebFetch）
- **一句话说明：** Skild 宣布 **S1 级基础模型** 可在 **NVIDIA Isaac Sim** 中通过 **单一得分目标 + 与近期自博弈** 后训练，涌现运球、护球、铲球等动态灵巧技能，并在 **约 140 年仿真对局** 后 **Sim2Real 迁移** 到人形真机足球对抗；定位为 Skild Brain 框架中 **ICL 预训练之后的 post-training 阶段**。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-09-24） |
|----|-------------------------|
| 项目页 | **无**独立研究项目页或代码仓链接；入口为公司博客 + [skild.ai](https://www.skild.ai/) |
| 代码 / 权重 | **确认未开源** — [github.com/skild-ai](https://github.com/skild-ai) **0 个公开仓**；正文未列 Hugging Face / 训练推理入口 |
| 仿真栈 | 文中点名 **NVIDIA Isaac Sim**；无公开环境 / 任务定义 / 奖励实现 |
| 数据集 | **未公开** |
| 可信度边界 | 产业官方博客 + 演示视频；无 peer-reviewed 论文、无可核对 benchmark 表 |

## 核心摘录（归纳，非全文）

### Skild Brain 框架位置

- **S1（上月博客）：** 预训练 + **in-context learning** — 人视频、手套、仿真、遥操作；任务由上下文示范指定。
- **本篇（post-training）：** **Self-improvement via reinforcement learning** — 部署或仿真中通过与 **近期版本自身** 对抗自我改进。
- 作者主张：S1 类模型受 **人类数据上限** 约束；**physical self-play** 是突破人类能力天花板的路径。

### 自博弈历史与动机

- 数字 AI：AlphaGo（2016）→ AlphaGo Zero（3 天自博弈 100:0）→ AlphaStar / OpenAI Five（2019）。
- 近年 **RL from verifiable rewards（RLVR）** 更简单可落地，自博弈热度下降；Skild 希望将其 **复活为 physical AGI 的点火器**。

### 方法与涌现行为

- **唯一显式目标：** score（得分）。
- **对手：** 策略的 **recent versions**；能力上升 → 对手同步变强。
- **仿真：** Isaac Sim 内 **数月～140 年** 量级对局（作者叙事时间线）。
- **能力曲线（定性）：** 初期几乎不会走 → 「大学年龄」可倒地起身 → 涌现 **dribbling / shielding / tackling**（无手工 shaping 奖励）。
- **Sim2Real：** 140 年仿真后策略迁入真机并完成对抗演示。

### 为何足球 & 扩展方向

- 现有 robot soccer cup 仍远低于人类；足球同时考验 **physical + strategic** 能力。
- 方法声称 **可泛化** 到日常机器人任务（非仅体育）。
- **At Scale 预告：** 更长仿真、虚拟工地/工厂/家庭；多智能体团队 → 协作操纵、城市级导航等 **社会行为** 涌现（下一篇 release）。

## 对 wiki 的映射

| 摘录主题 | 建议 wiki 落点 |
|----------|----------------|
| Skild Brain 三阶段 | [`skild-ai`](../entities/skild-ai.md)、[`skild-s1`](../entities/skild-s1.md)、新建 `skild-physical-self-play` |
| 自博弈 RL | [`reinforcement-learning`](../methods/reinforcement-learning.md)、[`deep-rl-game-milestones`](../concepts/deep-rl-game-milestones.md) |
| Isaac Sim + Sim2Real | [`sim2real`](../concepts/sim2real.md)、[`isaac-sim`](../entities/isaac-sim.md) |
| 人形动态对抗 | [`loco-manipulation`](../tasks/loco-manipulation.md)、[`paper-notebook-robostriker`](../entities/paper-notebook-robostriker.md) |
