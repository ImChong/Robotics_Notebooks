---
type: entity
tags: [curated-list, sim2real, real2sim, real2sim2real, domain-randomization, gaussian-splatting, embodied-ai]
status: complete
updated: 2026-09-04
related:
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../overview/hub-sim2real.md
  - ../concepts/sim2real.md
  - ../comparisons/sim2real-approaches.md
  - ../comparisons/sim2real-vs-real2sim-fine-tuning.md
  - ../methods/crisp-real2sim.md
  - ../queries/sim2real-gap-reduction.md
  - ../queries/sim2real-checklist.md
  - ./paper-agentic-real2sim.md
  - ./paper-simfoundry-real2sim-scene-generation.md
  - ./paper-arcadia.md
  - ./paper-r2s-ego.md
  - ./paper-lucida-r2s.md
  - ./awesome-world-models.md
  - ./awesome-touch.md
sources:
  - ../../sources/repos/awesome-real2sim2real.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/papers/arcadia_arxiv_2512_00076.md
summary: "sun254667 维护的 Awesome-Real2Sim2Real：按 Sim2Real → Real2Sim → Real2Sim2Real 闭环组织的迁移论文精选集；站内已节点化为技术地图 + paper-sa 详情页。"
---

# Awesome-Real2Sim2Real（sun254667 精选集）

**Awesome-Real2Sim2Real**（GitHub：[`sun254667/Awesome-Real2Sim2Real`](https://github.com/sun254667/Awesome-Real2Sim2Real)）是一份追踪 **Sim-to-Real / Real-to-Sim / Real2Sim2Real** 三线交织的 curated 论文列表：用真机数据修仿真、再用更好的仿真训更强真机策略，形成闭环叙事。

## 一句话定义

**迁移闭环** 文献索引入口：从经典 Sim2Real，到视频/操作 Real2Sim，再到 3DGS 等 Real2Sim2Real 管线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 用真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机的闭环迁移 |
| DR | Domain Randomization | 随机化仿真参数扩宽训练分布 |
| 3DGS | 3D Gaussian Splatting | 常见 Real2Sim 视觉重建表示 |

## 为什么重要

- **把「修仿真」与「修策略」放在同一地图**：避免只谈 DR/零样本而忽略 Real2Sim 资产侧。
- **与站内选型页对齐**：[Sim2Real](../concepts/sim2real.md)、[残差适配 vs Real2Sim vs 真机 RL](../comparisons/sim2real-vs-real2sim-fine-tuning.md)、[CRISP](../methods/crisp-real2sim.md) 可直接对照清单分节。
- **覆盖基础模型增强迁移**：Cosmos / 生成式 3D 世界等与 [Awesome World Models](./awesome-world-models.md) 交叉。

## 站内节点化

- **技术地图：** [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md)
- **目录 source：** [sun_awesome_r2s2r_catalog.md](../../sources/papers/sun_awesome_r2s2r_catalog.md)
- 新建索引级实体 `paper-sa-*`；已有同 arXiv canonical `paper-*` 则复用。

## 核心结构（怎么读）

| 区块 | 内容侧重 |
|------|----------|
| Surveys | MDP 视角 Sim2Real taxonomy、物理仿真器角色、reality gap 综述 |
| Sim-to-Real | RL 迁移 / 零样本 / DR·适配 / 基础模型增强 |
| Real-to-Sim | 视频重建（含 CRISP）/ 操作与交互场景 |
| Real2Sim2Real | 3DGS 框架与仿真管线闭环 |
| Benchmarks | 仿真器、数据集与评测协议 |

## 局限与使用注意

- **新建清单、星标仍低**：结构完整但条目密度会快速变化；入库后宜定期对照 upstream。
- **非工程 Runbook**：部署清单仍以站内 [Sim2Real Checklist](../queries/sim2real-checklist.md) / [闭环误差分层](../queries/sim2real-closed-loop-engineering.md) 为准。
- **开源状态逐条核**：论文宣称 code 不等于项目页已挂链接。

## 关联页面

- [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) — 清单论文 → 独立详情节点
- [Sim2Real（知识链汇总）](../overview/hub-sim2real.md) / [Sim2Real（概念）](../concepts/sim2real.md)
- [Sim2Real Approaches](../comparisons/sim2real-approaches.md)
- [Sim2Real 残差适配 vs Real2Sim vs 真机 RL](../comparisons/sim2real-vs-real2sim-fine-tuning.md)
- [CRISP Real2Sim](../methods/crisp-real2sim.md)
- [如何缩小 sim2real gap](../queries/sim2real-gap-reduction.md) / [Sim2Real Checklist](../queries/sim2real-checklist.md)
- [Agentic Real2Sim](./paper-agentic-real2sim.md) / [SimFoundry](./paper-simfoundry-real2sim-scene-generation.md) / [R2S-EGO](./paper-r2s-ego.md) / [Lucida](./paper-lucida-r2s.md)
- [Arcadia](./paper-arcadia.md) — 采集 + 生成式 USD + 共享 VLN/VLA + 反馈写回（部分开源）
- [Awesome World Models](./awesome-world-models.md) / [Awesome Touch](./awesome-touch.md)

## 参考来源

- [sources/repos/awesome-real2sim2real.md](../../sources/repos/awesome-real2sim2real.md)
- [sources/papers/sun_awesome_r2s2r_catalog.md](../../sources/papers/sun_awesome_r2s2r_catalog.md)

## 推荐继续阅读

- [GitHub 仓库 README](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [A Survey of Sim-to-Real Methods in RL（arXiv:2502.13187）](https://arxiv.org/abs/2502.13187)
- [CRISP（arXiv:2512.14696）](https://arxiv.org/abs/2512.14696) — 清单内代表性 Real2Sim 条目
