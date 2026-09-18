# A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models

> 来源归档（LongchaoDa 配套综述 · arXiv v3）

- **标题：** A Survey of Sim-to-Real Methods in RL: Progress, Prospects and Challenges with Foundation Models
- **作者：** Longchao Da, Justin Turnau, Thirulogasankar Pranav Kutralingam, Alvaro Velasquez, Paulo Shakarian, Hua Wei
- **arXiv：** 2502.13187（v3 修订 2025-03-08）
- **论文：** <https://arxiv.org/abs/2502.13187v3>
- **配套列表：** <https://github.com/LongchaoDa/AwesomeSim2Real>
- **入库日期：** 2026-09-18
- **一句话说明：** 首篇按 MDP 四要素（State / Action / Transition / Reward）系统梳理 Sim2Real RL 技法的综述，覆盖经典到基础模型增强路线，并总结评测流程与挑战。
- **沉淀到 wiki：** [`wiki/entities/paper-survey-sim2real-rl-foundation-models.md`](../../wiki/entities/paper-survey-sim2real-rl-foundation-models.md)

---

## 核心摘录（对 wiki 的映射）

1. **MDP 四要素 taxonomy** — State（观测 gap：DR、域适配、传感器融合、FM 视觉先验）、Action（尺度/延迟/不确定性）、Transition（动力学 gap：DR、Grounded Action Transformation、系统辨识）、Reward（ shaping / LLM 奖励设计）。→ 综述实体页「核心机制」+ [lc-awesome-sim2real 技术地图](../../wiki/overview/lc-awesome-sim2real-technology-map.md) 分组对齐。
2. **跨领域 Sim2Real** — 机器人、交通、推荐系统等；各域 simulators / benchmarks 分表汇总。→ 列表实体 [awesome-sim2real](../../wiki/entities/awesome-sim2real.md) 与 catalog。
3. **Foundation Models 视角** — VLM/LLM 用于观测对齐、奖励设计、语言条件策略与 sim2real prompt。→ 交叉 [generative-world-models](../../wiki/methods/generative-world-models.md)、[VLA](../../wiki/methods/vla.md)。
4. **形式化评测流程** — 强调可复现 benchmark / 开源代码清单；维护 AwesomeSim2Real 持续更新。→ 与 [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md) 工程选型互补。
5. **挑战与机遇** — reality gap 仍为核心瓶颈；FM 带来新工具但也引入新分布偏移。→ [Sim2Real Checklist](../../wiki/queries/sim2real-checklist.md)。

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| arXiv PDF + 配套 Awesome 列表 | 综述本身无训练代码 |
| 列表内逐条代码 | 以各论文项目页为准；列表 GitHub badge 仅覆盖部分条目 |

截至入库日：**已开源（导航级）** — 仓库 `LongchaoDa/AwesomeSim2Real` 为 Markdown 策展；**非** 可运行训练框架。
