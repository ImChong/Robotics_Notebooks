# orayyan.com/fetchman（FetchMan 项目页）

> 来源归档（ingest）

- **标题：** FetchMan: Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences
- **类型：** site / project-page
- **官方入口：** <https://orayyan.com/fetchman>
- **论文：** <https://arxiv.org/abs/2608.17027>
- **入库日期：** 2026-08-20
- **再核日期：** 2026-08-25
- **一句话说明：** UCLA 等人形 loco-manip 项目页：150k MolmoSpaces 场景合成数据、BC+Flow-GRPO 两阶段、G1 零样本视频与架构/消融表。

## 开源核查（步骤 2.5）

| 资源 | 2026-08-20 | 2026-08-25 再核 |
|------|------------|----------------|
| 项目首页 | 可访问；含 demo 视频、Table 1–2、Citation | 同左；页头新增 **Data & Code** |
| GitHub | **未列出** | [omarrayyann/fetchman](https://github.com/omarrayyann/fetchman) — **仅占位 README** |
| HF / 权重 | **未列出** | 仍 **未列出** |
| FetchMan-Bench | 论文与页面宣称发布；**无下载链** | 仍 **无下载链** |

**结论：** **部分开源 / 待发布** — 官方仓与项目页互链；README 写 **Code will be added by September 1**；训练/推理/Bench 截至再核日不可运行。

## 页面公开信息摘录

- **管线：** Stage 1 BC（DINOv3 + DiT delta chunk）→ Stage 2 Flow-GRPO（64×8 组、稀疏 grasp 奖励）。
- **数字（页面 Table）：** BC sim manip/loco-manip 75.0/67.0%；BC+RL 79.0/83.0%；真机 loco-manip 56.7%→73.3%。
- **消融：** SigLIP 或 absolute 动作 → 真机 loco-manip 0%。
- **致谢：** Nirvana 提供 G1；Mahi Shafiullah 讨论。

## 对 wiki 的映射

- [`wiki/entities/paper-fetchman.md`](../../wiki/entities/paper-fetchman.md)
- [`sources/papers/fetchman_arxiv_2608_17027.md`](../papers/fetchman_arxiv_2608_17027.md)
- [`sources/repos/fetchman.md`](../repos/fetchman.md)
