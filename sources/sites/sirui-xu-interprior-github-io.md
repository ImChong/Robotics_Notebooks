# sirui-xu.github.io/InterPrior（InterPrior 项目页）

> 来源归档（ingest）

- **标题：** InterPrior — UIUC × Amazon
- **类型：** site / project-page
- **官方入口：** <https://sirui-xu.github.io/InterPrior/>
- **入库日期：** 2026-05-17
- **一句话说明：** 论文配套站点：强调 **稀疏目标（球体等）下无稠密参考** 仍生成 **物理一致全身 loco-manipulation**；罗列 **G1 sim-to-sim、用户交互、多物体、同目标多样执行、扰动鲁棒、长时域随机切目标、失败恢复与起身** 等演示条；给出 **CVPR 2026 Highlight** 标签与 **BibTeX**；并链到 **InterMimic / InterAct / ULTRA / Dexplore** 等同系列工作。

## 页面公开信息（检索自 2026-05-17）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://sirui-xu.github.io/InterPrior/> |
| 论文 abs | <https://arxiv.org/abs/2602.06035> |
| PDF | <https://arxiv.org/pdf/2602.06035> |

## 与论文 HTML 版一致的公开主张（便于 wiki 溯源）

1. **核心叙事**：用 **大规模模仿预训练** 建立 **生成式 HOI 控制器**，再用 **RL 后训练** 在 **未见目标与初始化** 上巩固为可泛化 **运动先验**。
2. **能力展示（站点小节标题）**：Catch future snapshots / Follow trajectories / Capture contact（稀疏目标以球体可视化）；另含 multi-object、同目标多样执行、强鲁棒、长时程随机切换目标、失败恢复与起身等。
3. **基线对比条目标识**：对比 **InterMimic + MaskedMimic** 一类组合（以页面视频为准，定量以论文为准）。
4. **同系列索引**：页面列出 **ULTRA、Dexplore、InterMimic、InterAct、InterDreamer、InterDiff** 的入口链接，便于读者定位 **Inter-line** 研究谱系。

## 对 wiki 的映射

- [`wiki/entities/paper-interprior.md`](../../wiki/entities/paper-interprior.md) — 方法栈与局限归纳页。
- [`wiki/tasks/loco-manipulation.md`](../../wiki/tasks/loco-manipulation.md) — 任务语境（全身移动操作 + 物体动力学）。

## BibTeX（站点页提供，便于引用）

站点示例使用 `@article{xu2026interprior, ... eprint={2602.06035}, ...}`；正式引用以 arXiv / 录用版本为准。
