# NVIDIA 公司路线更新与健康度实测（2026-11-17）

范围为 `company.html?id=nvidia`，不调整评分函数、status 封顶或门禁。

- 时间线：31 → 36 个事件，35 个唯一详情节点（Isaac Lab 初版与 3.0 GA 复用同页）。
- 既有 31 个节点中，满分 17 → 22；新增事件复用四个已有满分详情节点，更新后唯一详情中 26/35 为 3/3。
- 本次改写的五个 wiki 页均为 3/3；但“路线全部满分”尚未完成，PR 保持草稿。
- 补充来源：[官方发布核查](../../sources/blogs/nvidia_robotics_latest_2026-11-17.md)。代码仓固定 commit 见各 sources 归档；没有把源码核对说成运行复现。

## 已完成

| 页面 | 健康度 | 缺项 |
|---|---|---|
| wiki/entities/isaac-gym.md | 3/3 | — |
| wiki/entities/curobo.md | 3/3 | — |
| wiki/entities/cosmos-curator.md | 3/3 | — |
| wiki/entities/cosmos-cookbook.md | 3/3 | — |
| wiki/entities/paper-bfm-14-hover.md | 3/3 | — |

## 待完成：原文与源码复核

以下为原有节点的实测缺项，不是本次改写页。下载到 PDF 不等于读完，不能只加“通读：已完成”或把 draft 改为 complete 以得到满分。需要逐页核对正文、必要附录、公开资产与源码入口，然后补齐结构和图表。当前不声称路线整体完成。

| 页面 | 健康度 | 缺项 |
|---|---|---|
| wiki/entities/mimicgen.md | 2/3 | 原文未通读、status 未 complete |
| wiki/entities/paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md | 1/3 | 阅读骨架、原文未通读 |
| wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md | 2/3 | 原文未通读 |
| wiki/entities/paper-doorman-opening-sim2real-door.md | 1/3 | 阅读骨架、原文未通读 |
| wiki/entities/paper-shenlan-wm-11-cosmos-policy.md | 1/3 | 阅读骨架、原文未通读 |
| wiki/entities/paper-hrl-stack-35-dreamdojo.md | 1/3 | 阅读骨架、原文未通读 |
| wiki/entities/kimodo.md | 1/3 | 阅读骨架、原文未通读 |
| wiki/entities/paper-motionbricks.md | 1/3 | 阅读骨架、原文未通读 |
| wiki/entities/paper-vesta-generalist-embodied-reasoning.md | 1/3 | 阅读骨架、原文未通读 |

完整复测：先按 company-roadmaps.json 的 NVIDIA 节点 ID 映射到 wiki，再用 scripts/report_health.py 对这些路径运行。评分、导出和 PR 正文门禁均沿用仓库原规则。
