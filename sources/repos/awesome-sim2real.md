# AwesomeSim2Real（LongchaoDa）

> 来源归档

- **标题：** AwesomeSim2Real
- **类型：** repo / awesome-list / curated-index
- **维护方：** [LongchaoDa](https://github.com/LongchaoDa)
- **链接：** <https://github.com/LongchaoDa/AwesomeSim2Real>
- **配套综述：** <https://arxiv.org/abs/2502.13187v3>（A Survey of Sim-to-Real Methods in RL, v3 2025-03-08）
- **规模（入库日）：** ~196★；README 内可解析论文链接约 139 条（83 条含 arXiv ID）
- **入库日期：** 2026-09-18
- **一句话说明：** RL 视角 Sim2Real 论文策展列表：按 MDP 四要素（State / Action / Transition / Reward）与领域（机器人、交通、推荐等）分类，配套 2025 综述与持续更新 issue 入口。
- **沉淀到 wiki：** [`wiki/entities/awesome-sim2real.md`](../../wiki/entities/awesome-sim2real.md)
- **技术地图 / 论文节点：** [`wiki/overview/lc-awesome-sim2real-technology-map.md`](../../wiki/overview/lc-awesome-sim2real-technology-map.md) · [`sources/papers/lc_awesome_sim2real_catalog.md`](../papers/lc_awesome_sim2real_catalog.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 不适用 |
|--------|--------|
| Markdown 策展清单 + 统计图 | 训练/推理代码、模型权重（清单性质） |

清单为 **资源导航**；复现价值在于按 MDP 要素与领域选型 Sim2Real RL 文献，并与配套综述交叉阅读。

---

## 核心结构（结构级）

1. Surveys and Simulators（综述 + 各域环境/基准）
2. Technique Papers — **Observation**（域随机化、域适配、传感器融合、基础模型等）
3. Technique Papers — **Action**（动作空间、延迟、不确定性等）
4. Technique Papers — **Transition**（DR、域适配、Grounding、LLM 增强等）
5. Technique Papers — **Reward**（Reward shaping、LLM 奖励设计等）

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [awesome-sim2real](../../wiki/entities/awesome-sim2real.md) | 本仓策展实体页 |
| [paper-survey-sim2real-rl-foundation-models](../../wiki/entities/paper-survey-sim2real-rl-foundation-models.md) | 配套综述深读页（arXiv:2502.13187v3） |
| [hub-sim2real](../../wiki/overview/hub-sim2real.md) / [sim2real](../../wiki/concepts/sim2real.md) | 站内 Sim2Real 知识链 |
| [sim2real-four-routes-identifiability](../../wiki/comparisons/sim2real-four-routes-identifiability.md) | 四条路线可辨识性对比（引用本综述） |
| [awesome-real2sim2real](awesome-real2sim2real.md) | 姊妹清单（sun254667 Real2Sim2Real 闭环，部分 arXiv 重叠） |
