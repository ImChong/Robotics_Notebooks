# Agentic Real2Sim Project Page

> 来源归档

- **标题：** Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents
- **类型：** site / project page
- **URL：** <https://agentic-real2sim.github.io/>
- **论文：** <https://arxiv.org/abs/2607.19190>
- **代码：** <https://github.com/agentic-real2sim/agentic_real2sim>（2026-11-16 核查已开放）
- **入库日期：** 2026-07-23；本次复核：2026-11-16
- **一句话说明：** 官方项目页展示将 **真机机器人–物体交互录像** 转为 **可仿真 episode twin** 的 Agentic Real2Sim：含 DROID↔MuJoCo 对照视频、浏览器 OpenUSD 孪生探索、可变形与人形适配示例，以及系统架构图。

## 开源状态（2026-11-16 官方页核查）

| 项 | 状态 |
|---|---|
| 论文 | [arXiv:2607.19190v5](https://arxiv.org/abs/2607.19190) |
| 代码 | **已发布**：[agentic-real2sim/agentic_real2sim](https://github.com/agentic-real2sim/agentic_real2sim) |
| CLI | `python -m ar2s.run_pipeline --raw-data ... --agent-config ...` |
| 复现依赖 | GPU、多环境、模型外部权重与部分 gated checkpoints；本次未实测 |
| 浏览器预览 | OpenUSD / three.js Hydra episode twins |
| 下游展示 | π0.5 策略评测、DROID-500 与六任务 sim/real 结果见 v5 论文 |

- [官方源码归档](../repos/agentic-real2sim.md) · [论文归档](../papers/agentic_real2sim_arxiv_2607_19190.md) · [实体节点](../../wiki/entities/paper-agentic-real2sim.md)

## 页面结构（策展）

- **Abstract** — 劳动密集 Real2Sim → VLM agent 编排物理世界建模
- **Explore Simulatable Episode Twins** — 浏览器内轨道/缩放浏览重建场景
- **Deformables / Humanoid** — PhysTwin 风格可变形；人形参考姿态 vs 仿真 settling
- **System Architecture** — 四代理流水线示意
- **Selected 25 Episodes / Video / BibTeX**

## 对 wiki 的映射

- 论文归档：[`sources/papers/agentic_real2sim_arxiv_2607_19190.md`](../papers/agentic_real2sim_arxiv_2607_19190.md)
- 沉淀 **[`wiki/entities/paper-agentic-real2sim.md`](../../wiki/entities/paper-agentic-real2sim.md)**
