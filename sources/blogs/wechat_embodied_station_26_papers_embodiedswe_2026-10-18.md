# 机器人会写控制程序之后，能否自己造训练数据？

> 来源归档（blog / 微信公众号）

- **标题：** 机器人会写控制程序之后，能否自己造训练数据？
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/mmnZuhiHkuzKCvdtU_XTNA
- **发表日期：** 2026-10-18
- **入库日期：** 2026-10-18
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md`](../raw/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)
- **一句话说明：** 机器人会写控制程序之后，成功程序本身就能变成可验证、可扩增的示范来源；与之配套的是用闭环任务而非数值误差来评测 VLA 的部署改动。本期深读 EmbodiedSWE，跟进 VLAQuantBench 与 Uranus。 共 26 篇；**26/26 独立详情节点**（本 ingest **新建 19**、**复用 7**）。

## 26 篇 → 本库节点

| # | 组 | 论文 | arXiv | 开源结论 | wiki |
|---|----|------|-------|----------|------|
| 01 | 数据生成与示范 | EmbodiedSWE | [2609.27308](https://arxiv.org/abs/2609.27308) | **已开源** | [paper-embodiedswe](../../wiki/entities/paper-embodiedswe.md)（**复用**） |
| 02 | VLA 评测与部署 | VLAQuantBench | [2609.25376](https://arxiv.org/abs/2609.25376) | **已开源** | [paper-vlaquantbench](../../wiki/entities/paper-vlaquantbench.md) |
| 03 | 世界模型与仿真 | Uranus | [2609.24815](https://arxiv.org/abs/2609.24815) | **已开源** | [paper-uranus](../../wiki/entities/paper-uranus.md)（**复用**） |
| 04 | 策略与规划 | HiRE | [2609.27068](https://arxiv.org/abs/2609.27068) | **已开源** | [paper-hire-hindsight-reward-editing](../../wiki/entities/paper-hire-hindsight-reward-editing.md) |
| 05 | VLA 评测与部署 | THAW-VLA | [2609.24682](https://arxiv.org/abs/2609.24682) | **已开源** | [paper-thaw-vla](../../wiki/entities/paper-thaw-vla.md)（**复用**） |
| 06 | 策略与规划 | X-Planner | [2609.25187](https://arxiv.org/abs/2609.25187) | **已开源** | [paper-x-planner-event-structured-task-planning](../../wiki/entities/paper-x-planner-event-structured-task-planning.md) |
| 07 | 世界模型与仿真 | H2RBench | [2609.24778](https://arxiv.org/abs/2609.24778) | **已开源** | [paper-h2rbench](../../wiki/entities/paper-h2rbench.md)（**复用**） |
| 08 | 世界模型与仿真 | D-JEPA | [2609.24749](https://arxiv.org/abs/2609.24749) | **已开源** | [paper-d-jepa-decision-aligned-latent-world-model](../../wiki/entities/paper-d-jepa-decision-aligned-latent-world-model.md) |
| 09 | VLA 评测与部署 | Gaussian Is Enough | [2609.27070](https://arxiv.org/abs/2609.27070) | **已开源** | [paper-gaussian-is-enough-flow-matching-priors](../../wiki/entities/paper-gaussian-is-enough-flow-matching-priors.md) |
| 10 | VLA 评测与部署 | LIBERO-VPro | [2609.24350](https://arxiv.org/abs/2609.24350) | **项目页未列 GitHub** | [paper-libero-vpro](../../wiki/entities/paper-libero-vpro.md) |
| 11 | 数据生成与示范 | Touch2Robot | [2609.24660](https://arxiv.org/abs/2609.24660) | **项目页未列 GitHub** | [paper-touch2robot](../../wiki/entities/paper-touch2robot.md) |
| 12 | 数据生成与示范 | ARSTAG | [2609.24563](https://arxiv.org/abs/2609.24563) | **已开源** | [paper-arstag](../../wiki/entities/paper-arstag.md) |
| 13 | 灵巧与触觉 | Contact-Anchored Retargeting | [2609.24093](https://arxiv.org/abs/2609.24093) | **已开源** | [paper-contact-anchored-retargeting-residual-dexterous](../../wiki/entities/paper-contact-anchored-retargeting-residual-dexterous.md) |
| 14 | 策略与规划 | PackLab | [2609.23784](https://arxiv.org/abs/2609.23784) | **已开源** | [paper-packlab](../../wiki/entities/paper-packlab.md) |
| 15 | VLA 评测与部署 | vla.simd | [2609.24274](https://arxiv.org/abs/2609.24274) | **项目页未列 GitHub** | [paper-vla-simd](../../wiki/entities/paper-vla-simd.md) |
| 16 | 策略与规划 | Visuomotor Pruning | [2609.24906](https://arxiv.org/abs/2609.24906) | **已开源** | [paper-visuomotor-robotic-pruning-orchards](../../wiki/entities/paper-visuomotor-robotic-pruning-orchards.md) |
| 17 | 数据生成与示范 | GLIDE | [2609.24996](https://arxiv.org/abs/2609.24996) | **项目页未列 GitHub** | [paper-glide-beyond-human-demonstration](../../wiki/entities/paper-glide-beyond-human-demonstration.md) |
| 18 | 策略与规划 | MIGU | [2609.24995](https://arxiv.org/abs/2609.24995) | **待发布** | [paper-migu-multimodal-instruction-grounding](../../wiki/entities/paper-migu-multimodal-instruction-grounding.md) |
| 19 | VLA 评测与部署 | RoboMP-DINOv2 | [2609.25506](https://arxiv.org/abs/2609.25506) | **已开源** | [paper-robomp-dinov2](../../wiki/entities/paper-robomp-dinov2.md) |
| 20 | 灵巧与触觉 | Berkeley QUAD Hand | [2609.27240](https://arxiv.org/abs/2609.27240) | **项目页未列 GitHub** | [paper-berkeley-quad-hand](../../wiki/entities/paper-berkeley-quad-hand.md) |
| 21 | 人形控制 | HOTICE | [2609.25363](https://arxiv.org/abs/2609.25363) | **待发布** | [paper-hotice](../../wiki/entities/paper-hotice.md)（**复用**） |
| 22 | 人形控制 | DeCap | [2609.24552](https://arxiv.org/abs/2609.24552) | **已开源** | [paper-smoothness-constraint-locomotion](../../wiki/entities/paper-smoothness-constraint-locomotion.md)（**复用**） |
| 23 | VLA 评测与部署 | Topology-VLA | [2609.23944](https://arxiv.org/abs/2609.23944) | **项目页未列 GitHub** | [paper-topology-informed-visual-prompting-vla](../../wiki/entities/paper-topology-informed-visual-prompting-vla.md) |
| 24 | 人形控制 | PRIMO | [2609.23610](https://arxiv.org/abs/2609.23610) | **已开源** | [paper-primo-human-motion-odometry](../../wiki/entities/paper-primo-human-motion-odometry.md)（**复用**） |
| 25 | 灵巧与触觉 | SpectRobot | [2609.24621](https://arxiv.org/abs/2609.24621) | **项目页未列 GitHub** | [paper-spectrobot-single-point-tactile](../../wiki/entities/paper-spectrobot-single-point-tactile.md) |
| 26 | 灵巧与触觉 | Norm2Tex | [2609.25398](https://arxiv.org/abs/2609.25398) | **待发布** | [paper-norm2tex](../../wiki/entities/paper-norm2tex.md) |

## 对 wiki 的映射

- **26/26 独立详情节点**；同 arXiv 不重复造页（复用节点已补本文来源与地图关联）。
- 阅读坐标：[编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
- 开源结论口径：GitHub 仓库用 `git ls-remote` 核实可达；项目页用 curl 读取扫描 GitHub 链接与 "coming soon"；「项目页未列 GitHub」表示入库日页面无代码链接且未见开源承诺字样。

## 当前提炼状态

- [x] 公众号正文抓取（2026-10-18）
- [x] 26 篇独立节点（19 新建 / 7 复用）
- [x] 项目页 / 仓库开源状态核查（步骤 2.5）
- [ ] 原文实验表格核验（节点内数字仅为公众号转述）
