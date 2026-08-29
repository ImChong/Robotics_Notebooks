# Riemann-1.0 项目页

> 来源归档（ingest）

- **标题：** Riemann-1.0 | Advanced Embodied World Action Model
- **类型：** site / project-page / company technical report
- **项目页：** <https://riemann-dynamics.github.io/Riemann-1.0-Website>
- **规范 URL（og:url）：** <https://riemann-1.github.io/>
- **论文 PDF：** <https://riemann-dynamics.github.io/Riemann-1.0-Website/paper/Riemann-1.0.pdf>
- **arXiv：** **无**（截至 2026-08-29 项目页仅托管 PDF，未列 arXiv）
- **代码：** **未开源**（页头仅 Paper；组织仓只有官网静态站，无训练/推理/权重）
- **数据集：** **未公开**
- **机构：** 黎曼动力（Riemann Dynamics）；昆仑万维（Kunlun Wanwei）子公司
- **入库日期：** 2026-08-29
- **一句话说明：** 黎曼动力官方 Riemann-1.0 项目页：全因果自回归 World Action Model，统一可执行策略与动作条件世界仿真；232K+ 小时异构具身数据，仿真与天机 Marvin 真机成绩。

## 开源核查（步骤 2.5，2026-08-29）

| 项 | 状态 |
|----|------|
| 项目页导航 | 仅 **Paper** → 站点内 PDF；无 Code / Hugging Face / ModelScope |
| GitHub `Riemann-Dynamics` | 公开仓 2 个：`Riemann-1.0-Website`（本站静态文件）、`Matrix-Game-3.5`（交互式世界模型，**非** Riemann-1.0） |
| 训练 / 推理 / 权重 / 数据 | **确认未开源** |
| 结论 | **闭源产业技术报告**；可作因果 WAM 选型与榜数对照，不可复现训练栈 |

## 页面结构（策展）

| 区块 | 内容要点 |
|------|----------|
| Overview | 全因果 AR WAM；三阶段渐进预训练；策略 / 仿真双接口 |
| Real-World | 天机 Marvin 四任务：Desk / Clothes / Cube stacking / Kitchen；组合泛化与 OOD |
| Simulation | LIBERO 99.0、RoboTwin 2.0 94.3、RoboCasa365 62.6 |
| Data Infra | 232K+ h；VLM 分层切段、3D 手重建、场景–技能平衡 |
| Method | 共享 Video/Action DiT + 结构化因果 mask；λ=0.1 / 0.5 / 0.9 |
| Inference | 动作 chunk 去噪 + 真观测写回 cache；World Simulator 演示 |

## 对 wiki 的映射

- 论文归档：[`sources/papers/riemann_1_0.md`](../papers/riemann_1_0.md)
- 官网仓：[`sources/repos/riemann-1-0-website.md`](../repos/riemann-1-0-website.md)
- 沉淀实体：[`wiki/entities/paper-riemann-1.md`](../../wiki/entities/paper-riemann-1.md)
