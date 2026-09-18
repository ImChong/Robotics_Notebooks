# HumanCLAW（VLM 能否通过身体行动？）

> 来源归档（ingest）

- **标题：** HumanCLAW: Can Vision-Language Models Act Through a Body?
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2607.27180>
- **项目页：** <https://human-claw.github.io/>
- **代码：** <https://github.com/Human-CLAW/HumanCLAW>
- **Motion 权重：** <https://huggingface.co/HumanCLAW/HumanCLAW>
- **Benchmark 数据补充：** <https://huggingface.co/datasets/HumanCLAW/HumanCLAW-HSSD>
- **机构：** Meta；南洋理工大学（NTU）；华盛顿大学（UW）；布朗大学（Brown）；西北大学（Northwestern）
- **入库日期：** 2026-09-18
- **一句话说明：** 提出 **action intelligence** 评测框架 HumanCLAW 与 **HumanCLAW-Bench**（1,218 条 HSSD 室内 egocentric find–navigate–interact 回合），解耦 VLM 决策与全身运动执行；九款冻结 SOTA VLM 最高 InteractSR 仅 **16.8%**，暴露 embodied self-awareness 缺口。

## 核心摘录（策展）

### 1) Action intelligence 与三角色解耦

- **摘录要点：** **Action intelligence** 指在物理闭环中每时刻决定「身体下一步执行什么」，区别于 motor control。HumanCLAW 解耦：（1）**VLM skill harness** — 每 0.5 s 读 egocentric 视图，三阶段推理（visual state → 2–3 s mid-level goal → 原子技能+参数）；（2）**verifier** — 短上下文 spatial-action outcome 校验；（3）**motion generator** — DiT 人体先验 + 每技能 ControlNet adapter；（4）**half-physics simulator** — kinematic 执行 + 真实碰撞/重力/物体响应，排除 balance/tracking 失败。
- **对 wiki 的映射：**
  - [HumanCLAW](../../wiki/entities/paper-humanclaw.md) — 框架与流程图。
  - [VLA](../../wiki/methods/vla.md) — VLM 作为决策层而非端到端 VLA 的对照语境。

### 2) HumanCLAW-Bench 任务与指标

- **摘录要点：** Progressive `find–navigate–interact`：「find a `<category>`, navigate to it with zero distance, and finally sit on it」。41 validation houses → **1,218 episodes**；6 类目标；可动物体动态化以测 disturbance。**FindSR**（目标 ≥100 semantic px 且模型 acknowledge）、**NavSR**（≤20 cm 且 stopping）、**InteractSR**（pelvis contact sit）。另报 collision fraction、object disturbance、motion jerk、token cost；难度三维 stratification（distance / choice / obstacle）。
- **对 wiki 的映射：**
  - [HumanCLAW](../../wiki/entities/paper-humanclaw.md) — 评测表与读法。
  - [HumanoidArena](../../wiki/entities/paper-humanoidarena.md) — 另一 egocentric 全身 benchmark 对照。

### 3) 九 VLM leaderboard 与六条 findings

- **摘录要点：** 冻结 off-the-shelf VLM；最佳 **Gemini-3.1 InteractSR 16.8%**；GPT-6 FindSR 75.5% 但 NavSR 57.1%。Findings：（1）无模型 solve benchmark；（2）结构化 memory / mid-level objective / verifier 有效，单纯加长 history 饱和且 10 帧 hurt；（3）识别非 finding 主瓶颈，探索不足占 38% failures；（4）egocentric self-localization 是 navigation 主瓶颈（68% active-find 仍 Nav 失败）；（5）reach≠interact（contact 后 success 90%→3.5% 跨模型）；（6）VLMs 缺乏 body awareness（腿/脚 collision 28–45% steps）。
- **对 wiki 的映射：**
  - [HumanCLAW](../../wiki/entities/paper-humanclaw.md) — 结论与工程含义。

### 4) 开源状态（截至 2026-09-18，项目页核查）

- **摘录要点：** **已开源** Apache 2.0 完整评测栈（2026-08-17）；HF 发布 motion weights；HSSD baked mesh **gated** 补充包；需自备授权 HSSD-Hab 与 VLM endpoint。Motion 训练实现与 per-skill chunk 列表含于仓库。
- **对 wiki 的映射：**
  - [humanclaw 仓库](../repos/humanclaw.md)
  - [human-claw 项目页](../sites/human-claw-github-io.md)

## 当前提炼状态

- [x] arXiv / 项目页 / GitHub / HF 已交叉核查
- [x] wiki 映射：`wiki/entities/paper-humanclaw.md`
