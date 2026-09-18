---
type: entity
tags: [curated-list, sim2real, reinforcement-learning, domain-randomization, foundation-models, survey, arizona-state]
status: complete
updated: 2026-09-18
related:
  - ../overview/lc-awesome-sim2real-technology-map.md
  - ../overview/hub-sim2real.md
  - ../concepts/sim2real.md
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../comparisons/sim2real-approaches.md
  - ./paper-survey-sim2real-rl-foundation-models.md
  - ./awesome-real2sim2real.md
  - ./paper-pace-sim2real-legged-robots.md
sources:
  - ../../sources/repos/awesome-sim2real.md
  - ../../sources/papers/lc_awesome_sim2real_catalog.md
  - ../../sources/papers/lc_awesome_sim2real_survey_2502_13187.md
summary: "LongchaoDa 维护的 AwesomeSim2Real：按 MDP 四要素与领域分类的 RL Sim2Real 论文策展库，配套 arXiv:2502.13187v3 综述；站内已节点化为技术地图 + paper-as 详情页。"
---

# AwesomeSim2Real（LongchaoDa）

**AwesomeSim2Real**（GitHub：[LongchaoDa/AwesomeSim2Real](https://github.com/LongchaoDa/AwesomeSim2Real)）是配套综述 [A Survey of Sim-to-Real Methods in RL（2502.13187v3）](./paper-survey-sim2real-rl-foundation-models.md) 的 **持续维护** 论文列表：按 **State / Action / Transition / Reward** 与 simulators/benchmarks 组织 RL Sim2Real 文献。

## 一句话定义

**RL Sim2Real 策展索引** — 用 MDP 四要素 taxonomy 导航经典到基础模型增强的迁移论文，并覆盖机器人、交通、推荐等多域 simulators。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机/真实环境 |
| MDP | Markov Decision Process | 状态–动作–转移–奖励形式化 |
| DR | Domain Randomization | 域随机化 |
| FM | Foundation Model | 大模型增强 Sim2Real |
| GAT | Grounded Action Transformation | 仿真转移 grounding |

## 为什么重要

- **与综述一体：** 列表结构与 [2502.13187 综述](./paper-survey-sim2real-rl-foundation-models.md) 章节对齐，读 taxonomy 可直接跳到代表论文。
- **跨域广：** 除机器人外含交通、推荐等，避免 Sim2Real 知识仅绑死在 manipulation/loco。
- **可贡献更新：** README 欢迎 issue 补充新文；站内 catalog 可通过脚本重生成。
- **与 sun254667 Real2Sim2Real 互补：** 后者偏 Real2Sim2Real 闭环与 3DGS；本列表偏 **RL + MDP 要素** 分类。

## 站内节点化

- **技术地图：** [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md)
- **目录 source：** [lc_awesome_sim2real_catalog.md](../../sources/papers/lc_awesome_sim2real_catalog.md)
- 新建清单索引页 `paper-as-*`；已有同 arXiv canonical `paper-*` 则复用（避免重复节点）。

## 核心结构（怎么读）

| 区块 | 内容侧重 |
|------|----------|
| Surveys & Simulators | 历史综述 + 各域 Environment / Sim2Real Benchmark |
| Observation | DR、DA、传感器融合、Foundation Models |
| Action | 动作尺度、延迟、不确定性 |
| Transition | DR、GAT、LLM 增强转移 |
| Reward | Shaping、LLM 奖励设计 |

## 局限与使用注意

- **索引级节点为主：** 大部分 `paper-as-*` 为策展摘录；主线论文应升格深度页（如 [PACE](./paper-pace-sim2real-legged-robots.md)）。
- **非工程 Runbook：** 部署清单见 [Sim2Real Checklist](../queries/sim2real-checklist.md)。
- **开源逐条核：** GitHub badge 仅覆盖部分条目；复现前打开论文项目页。

## 关联页面

- [AwesomeSim2Real 技术地图](../overview/lc-awesome-sim2real-technology-map.md) — 清单论文 → 独立详情节点
- [Sim2Real RL 综述（2502.13187）](./paper-survey-sim2real-rl-foundation-models.md)
- [Sim2Real Hub](../overview/hub-sim2real.md) / [Sim2Real 概念](../concepts/sim2real.md)
- [Sim2Real 四条路线](../comparisons/sim2real-four-routes-identifiability.md)
- [Awesome-Real2Sim2Real](./awesome-real2sim2real.md) — Real2Sim2Real 闭环姊妹清单

## 参考来源

- [sources/repos/awesome-sim2real.md](../../sources/repos/awesome-sim2real.md)
- [lc_awesome_sim2real_catalog.md](../../sources/papers/lc_awesome_sim2real_catalog.md)
- [lc_awesome_sim2real_survey_2502_13187.md](../../sources/papers/lc_awesome_sim2real_survey_2502_13187.md)

## 推荐继续阅读

- [GitHub 仓库 README](https://github.com/LongchaoDa/AwesomeSim2Real)
- [综述 arXiv:2502.13187v3](https://arxiv.org/abs/2502.13187v3)
