# 具身智能资源合集：代码、评测工具与待开源项目一次整理

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能资源合集：代码、评测工具与待开源项目一次整理
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/lnsitff1SA3xPNj5tDwxPQ
- **发表日期：** 2026-09-15
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch（公众号正文）
- **一句话说明：** 汇总 9 篇具身论文（潜动作世界模型、物理基础模型、材料自动化、跨具身 WBC、语言触觉、多机工厂导航、情绪步态、软体 aSSM、双臂 one-shot）+ 用户补充 **EffVLA** 设计空间研究；**10/10 均有独立 `paper-*` 详情节点**（本 ingest **新建 9**；PhysBrain 1.5 **复用**既有实体；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：本期覆盖跨具身控制、VLA/物理基础模型、触觉与材料识别、软体机器人与多机器人自动化；重点推荐 ACT-LAM 与 PhysBrain 1.5。

### 10 篇 → 本库节点

| # | 论文 | arXiv / 发布 | 开源结论（入库日） | wiki |
|---|------|-------------|-------------------|------|
| 01 | ACT-LAM | [2609.15189](https://arxiv.org/abs/2609.15189) | **已开源** `DingjieFu/ACT-LAM` | [paper-act-lam](../../wiki/entities/paper-act-lam.md) |
| 02 | PhysBrain 1.5 | [2609.14973](https://arxiv.org/abs/2609.14973) | **部分开源** HF 权重 + PhysBrainEvalKit | [paper-sa-2512-16793-physbrain…](../../wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md)（**复用**） |
| 03 | SAIGEN | [2609.14928](https://arxiv.org/abs/2609.14928) | **已开源** `YusukeHashimotoLab/saigen` | [paper-saigen](../../wiki/entities/paper-saigen.md) |
| 04 | X-WBC | [2609.15213](https://arxiv.org/abs/2609.15213) | **已开源** `LogosRoboticsGroup/x-wbc`（CoRL 2026） | [paper-x-wbc](../../wiki/entities/paper-x-wbc.md) |
| 05 | Language-Guided Tactile | [2609.14783](https://arxiv.org/abs/2609.14783) | **已开源** `Mashood3624/Language_Tactile` | [paper-language-guided-tactile](../../wiki/entities/paper-language-guided-tactile.md) |
| 06 | FMAPPO | [2609.14567](https://arxiv.org/abs/2609.14567) | **待发布**：仅匿名项目页，无 GitHub | [paper-fmappo](../../wiki/entities/paper-fmappo.md) |
| 07 | EMoG | [2609.14432](https://arxiv.org/abs/2609.14432) | **待发布**：仅项目页，无 GitHub | [paper-emog](../../wiki/entities/paper-emog.md) |
| 08 | aSSMPy / aSSM 姿态控制 | [2609.14376](https://arxiv.org/abs/2609.14376) | **待核实**：文内链 `karakaron/aSSMPy` 入库日 **404** | [paper-assmpy-soft-robot-orientation](../../wiki/entities/paper-assmpy-soft-robot-orientation.md) |
| 09 | VLBiMan++ | [2609.14310](https://arxiv.org/abs/2609.14310) | **已开源** `hnuzhy/BiRoMan` + 项目页 | [paper-vlbiman-plus](../../wiki/entities/paper-vlbiman-plus.md) |
| 10 | EffVLA | 项目页（arXiv 待挂） | **部分开源** `mindvla-team/EFFVLA` action-head 建模代码 | [paper-effvla](../../wiki/entities/paper-effvla.md) |

## 对 wiki 的映射

- **10/10 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 9** 个实体；**1 复用** PhysBrain 1.5；**0 重复 arXiv 节点**。
- 阅读坐标：[具身资源 10 篇技术地图](../../wiki/overview/embodied-resources-10-papers-technology-map.md)。
- 交叉：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Manipulation](../../wiki/tasks/manipulation.md)。

## 当前提炼状态

- [x] 公众号正文抓取
- [x] EffVLA 项目页核查（步骤 2.5）
- [x] 10 篇独立节点规划（9 新建 / 1 复用）
