# LIFT 项目页（lift-policy.github.io · Reactive Force VLA Post-Training）

> 来源归档

- **标题：** LIFT: Never Too Late for Force — Accelerating VLA Post-Training with Reactive Force Injection
- **类型：** site（项目页）
- **链接：** <https://lift-policy.github.io/>
- **论文：** <https://arxiv.org/abs/2607.14236>（CoRL 2026）
- **代码：** <https://github.com/y-wng/lift>
- **机构：** 上海交通大学、上海创智学院、南方科技大学、致远学院、诺玛矩阵
- **入库日期：** 2026-09-27
- **一句话说明：** 官方页：方法三块（reactive force injection / prior-preserving init / heterogeneous + online DAgger）、三任务学习曲线与 ablation、泛化 shift 图、任务视频与 BibTeX。

## 源码开放核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 代码 | **已开源** — 页内与 arXiv 链到 **`y-wng/lift`**（OpenPI 系训练/推理/online launcher） |
| 权重 | **未随仓发布** — 需自备 π₀.₅ 初始化 checkpoint 与 LeRobot 数据 |
| 数据 | **未发布** — 示范为 handheld vision-only + Flexiv TDK 纠错；格式说明在仓库 README |
| 真机栈 | **部分** — 仓内 **`serve_policy.py` + WebSocket client**；Flexiv 驱动、TDK、NEDF2 上传与 `nmx_nedf_api` **外部** |
| 判定 | **部分开源**（训练/推理管线可用；完整闭环依赖 Flexiv 生态与部署脚本） |

## 页面列出的核心对比（摘要）

- **LIFT vs π₀.₅ online DAgger（无力）：** 三任务学习曲线整体更高更快。
- **LIFT vs 单帧力：** 书/汉诺塔 reactive 历史优势明显；毛巾可接近。
- **LIFT vs offline DAgger：** offline 力 buffer 全任务落后，book insertion **0**。
- **LIFT vs residual policy：** residual 三任务显著更低。

## 相关归档

- [`sources/papers/lift_reactive_force_vla_arxiv_2607_14236.md`](../papers/lift_reactive_force_vla_arxiv_2607_14236.md)
- [`sources/repos/y-wng-lift.md`](../repos/y-wng-lift.md)
- 沉淀到 wiki：[`wiki/entities/paper-lift-reactive-force-vla-posttrain.md`](../../wiki/entities/paper-lift-reactive-force-vla-posttrain.md)

> **名称消歧：** 本 **LIFT** = Late Reactive Injection of Force（VLA 后训练）。库内 [`lift-humanoid.md`](../../wiki/entities/lift-humanoid.md) 为 BIGAI **人形 RL 预训练+微调**（arXiv:2601.21363），缩写相同、主题不同。
