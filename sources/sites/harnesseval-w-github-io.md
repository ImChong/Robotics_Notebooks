# HarnessEval-W（项目页）

> 来源归档（ingest 关联资料）

- **标题：** HarnessEval-W — Agentifying the Evaluation of Visual Worlds
- **类型：** site / project-page / leaderboard
- **项目页：** <https://mirros-lab.github.io/HarnessEval-W>
- **论文：** <https://arxiv.org/abs/2608.16859>
- **代码：** <https://github.com/mirros-lab/harnesseval-w> — 见 [`sources/repos/harnesseval-w.md`](../repos/harnesseval-w.md)
- **Blog：** <https://mirros.ai/blog/harnesseval> — 见 [`sources/blogs/mirros_harnesseval.md`](../blogs/mirros_harnesseval.md)
- **机构：** 镜界（MirroS）/ MirroS-Lab
- **入库日期：** 2026-08-18
- **一句话说明：** HarnessEval-W 官方入口：口号「Evaluation defines the taste of evolution」、覆盖数字、人对齐摘要、2026-08-18 V1 Leaderboard（Coming Soon）与 BibTeX。

## 开源核查（步骤 2.5，2026-08-18）

项目页头部 / Resources 明确链到 **Code**（GitHub）与论文；口号区写 open-sourced as an executable agentic system。

| 链接 | 用途 |
|------|------|
| GitHub `MirroS-Lab/HarnessEval-W` | 评测代码、plans、demo（README 宣称 Apache-2.0） |
| arXiv:2608.16859 | 论文 |
| mirros.ai/blog/harnesseval | 概念长文 |
| Hugging Face 全量案例 | 项目页未列；README TODO 仍待勾 |
| Leaderboard 2026-08-18 V1 | 页面写 **Coming Soon**；筛选器含 Prompt I2V / Camera pose / Native action |

**结论：** 代码入口以项目页实际 GitHub 链接为准 → **已开源评测系统**；活榜与 HF 数据卡截至入库日未上线。

## 项目页数字快照（2026-08-18）

| 项 | 数值 |
|----|------|
| 评测案例 | 330（108 exploratory、51 intentional、66 physical；34 drift、34 revisit、37 offscreen） |
| 技能数 | **11** specialized evaluation skills |
| 已打分 rollout | 5,940（含 planner→validation 完整推理迹） |
| 榜上模型 | 18 |
| Intentional vs 人类 BT | Spearman ρ = 0.93 |
| Physical 成对准确率 | 71.7%（对照最近 WBench 协议 31.9%） |
| 三次重复包络 | 比 WBench 窄 4.9× |

## 对 wiki 的映射

- [`wiki/entities/paper-harnesseval-w.md`](../../wiki/entities/paper-harnesseval-w.md) — 论文实体与选型读法
- [`sources/papers/harnesseval_w_arxiv_2608_16859.md`](../papers/harnesseval_w_arxiv_2608_16859.md) — 论文摘录
- [`sources/repos/harnesseval-w.md`](../repos/harnesseval-w.md) — 仓库入口
