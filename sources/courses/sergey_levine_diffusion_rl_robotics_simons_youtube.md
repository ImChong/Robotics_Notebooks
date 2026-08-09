# Diffusion in RL and robotics — Sergey Levine（Simons Institute / YouTube）

> 来源归档

- **标题：** Diffusion in RL and robotics: how expressive policies changed how we use continuous actions
- **类型：** course / video（学术研讨会录像）
- **讲者：** Sergey Levine（UC Berkeley）— <https://people.eecs.berkeley.edu/~svlevine/>
- **主办方 / 频道：** Simons Institute for the Theory of Computing — <https://www.youtube.com/@SimonsInstitute>
- **工作坊：** [Diffusion Generative Modeling: Progress and Next Steps](https://simons.berkeley.edu/workshops/diffusion-generative-modeling-progress-next-steps)（2026-08-03 – 2026-08-07，Calvin Lab）
- **官方 talk 页：** <https://simons.berkeley.edu/talks/sergey-levine-uc-berkeley-2026-08-07>
- **链接：** <https://www.youtube.com/watch?v=agi3xLTGyaU>（移动端同源：`m.youtube.com/watch?v=agi3xLTGyaU`）
- **Video ID：** `agi3xLTGyaU`
- **日程时段：** Friday, Aug. 7, 2026，9 – 9:45 a.m. PT（约 45 min 槽位）
- **直播日期：** 2026-08-07（频道标注 Streamed live）
- **入库日期：** 2026-08-09
- **一句话说明：** Levine 在 Simons「Diffusion Generative Modeling」工作坊的收官报告：扩散与 flow 把连续动作策略的表达能力抬高后，**大块 action chunk** 成为可行默认，从而推动 IL，并进一步改善 offline RL / offline-to-online RL 与大规模生成式控制模型。

## 为什么值得保留

- **一手议程坐标**：官方 abstract 明确把「表达力更强的动作分布 → action chunks → IL / offline RL」串成一条叙事，适合作为本库 [Diffusion Policy](../../wiki/methods/diffusion-policy.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[Online vs Offline RL](../../wiki/comparisons/online-vs-offline-rl.md) 的**当代讲者读法**。
- **跨范式桥接**：同一套生成式动作头既服务模仿学习，也被讲成 offline RL 与 offline-to-online 的近期进展载体——与 [LWD](../../wiki/methods/lwd.md)、flow matching VLA（如 [π₀](../../wiki/methods/π0-policy.md)）可对读。
- **工作坊语境**：该场次隶属扩散生成建模理论/实践预览会（Fall '27 学期预告），机器人侧听众可借此把控制应用挂回更广的 diffusion 数学议程。

## 抓取与字幕说明（入库日）

| 通道 | 结果 |
|------|------|
| **Agent Reach** | v1.5.0 已装；`doctor` 在配置 `yt-dlp --js-runtimes node` 后标 YouTube 渠道可用 |
| **yt-dlp 元数据 / 字幕** | 本机 IP 触发 YouTube「Sign in to confirm you’re not a bot」；多 client（android/web/ios）均失败 |
| **oEmbed / Jina Reader** | 可用：标题、频道、讲者与 talk 页链接、约 1.4K views（入库时） |
| **Simons talk 页** | **主内容来源**：日程、abstract、speaker 链接完整 |
| **结论** | 本条以 **官方 abstract + 日程元数据** 归纳；**非**字幕全文转写。后续若字幕可抽，应回填「章节结构 / Q&A」节 |

## 官方 Abstract（摘录并保留结构）

> In this talk, I'll discuss how diffusion and flow models transformed continuous-action policies in RL and robotic control. While in principle learning-based control methods (imitation learning and RL) are largely agnostic to the particular representation used for the action distribution, in practice diffusion and flow models have significantly improved the performance of learned policies across various learning-based control domains. By modeling complex distributions over high-dimensional spaces, these models made it possible to represent large "action chunks" (sequences of actions), which greatly improved imitation learning performance and, most recently, the performance of offline RL and offline-to-online RL methods. In this talk, I'll discuss algorithms that use diffusion and flow for RL and IL, as well as some of the large-scale models built on these principles.

## 核心观点（按 abstract 归纳，非字幕全文）

1. **表示在原理上可替换，在实践中不可忽视**：IL/RL 算法对动作分布族「名义上」中立，但扩散 / flow 作为动作头后，经验性能显著抬升。
2. **表达力解锁高维动作块**：复杂高维分布建模使 **长 action chunk（动作序列）** 可学、可采样，而不被迫塌缩为单步高斯均值。
3. **先抬 IL，再抬 offline RL**：chunk + 生成式动作头首先改善模仿学习；近期同样改善 **offline RL** 与 **offline-to-online RL**。
4. **议程覆盖**：扩散 / flow 用于 RL 与 IL 的算法，以及建于其上的 **大规模模型**（通才 / foundation 控制叙事的上游读法）。

## 章节结构（按 abstract 主题推断；非官方时间戳）

| 部分 | 主题（待字幕核对） |
|------|-------------------|
| 开场 | 连续控制里「动作分布表示」为何从理论中立变成实践瓶颈 |
| 机制 | 扩散 / flow 如何表达多模态与高维序列动作 |
| IL | action chunks 与 visuomotor / 操作模仿学习的收益 |
| RL | offline RL、offline-to-online 上的同族动作头 |
| 规模化 | 大模型 / 通才策略如何继承上述原则 |
| 收束 | 开放问题与和生成建模理论议程的接口 |

## 对 wiki 的映射

- [`wiki/overview/sergey-levine-diffusion-expressive-policies.md`](../../wiki/overview/sergey-levine-diffusion-expressive-policies.md) — **父节点**（本视频阅读坐标）
- [`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md) — 扩散动作头在 IL 中的方法页
- [`wiki/methods/action-chunking.md`](../../wiki/methods/action-chunking.md) — abstract 强调的「大块动作序列」机制
- [`wiki/concepts/diffusion-model.md`](../../wiki/concepts/diffusion-model.md) — 扩散生成式建模概念
- [`wiki/methods/imitation-learning.md`](../../wiki/methods/imitation-learning.md) — IL 主线
- [`wiki/comparisons/online-vs-offline-rl.md`](../../wiki/comparisons/online-vs-offline-rl.md) — offline / offline-to-online 对照
- [`wiki/methods/lwd.md`](../../wiki/methods/lwd.md) — 车队级 offline-to-online 实例（flow 动作头）
- [`wiki/methods/π0-policy.md`](../../wiki/methods/π0-policy.md) — flow matching 通才策略对照
- [`sources/sites/simons_sergey_levine_diffusion_rl_robotics_2026.md`](../sites/simons_sergey_levine_diffusion_rl_robotics_2026.md) — 官方 talk 页归档

## 推荐继续阅读（外部）

- [Simons talk 页](https://simons.berkeley.edu/talks/sergey-levine-uc-berkeley-2026-08-07)
- [工作坊主页](https://simons.berkeley.edu/workshops/diffusion-generative-modeling-progress-next-steps)
- [工作坊日程（含本场 Video 链）](https://simons.berkeley.edu/workshops/diffusion-generative-modeling-progress-next-steps/schedule)
- [Sergey Levine 主页 / RAIL](https://people.eecs.berkeley.edu/~svlevine/)
- [YouTube 录像](https://www.youtube.com/watch?v=agi3xLTGyaU)
