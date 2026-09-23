# STRIDER: Stepping-Enabled Multi-Gait Hierarchical 3D Loco-Manipulation Framework for Humanoid Robots

> 来源归档（ingest · 演示视频 · 尚无公开论文 PDF）

- **标题：** STRIDER: Stepping-Enabled Multi-Gait Hierarchical 3D Loco-Manipulation Framework for Humanoid Robots
- **类型：** video / humanoid / loco-manipulation / hierarchical-control / multi-gait / stepping
- **机构：** 北京人形机器人创新中心（X-Humanoid / Beijing Innovation Center of Humanoid Robotics）
- **演示视频：** <https://youtu.be/gf5RWjCZXtA> — 归档见 [`sources/sites/strider-demo-youtube.md`](../sites/strider-demo-youtube.md)
- **论文 / arXiv：** 截至 **2026-09-23** 入库日，**未检索到** 公开 arXiv 或 PDF；仅有 YouTube 非公开演示片
- **入库日期：** 2026-09-23
- **一句话说明：** 北京人形创新中心展示的 **分层 3D 移动操作** 框架：强调 **步态切换（stepping-enabled multi-gait）** 与 **全身 loco-manipulation** 的层级控制；当前公开材料仅为约 4 分 51 秒 demo 视频。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 演示视频 | <https://youtu.be/gf5RWjCZXtA> | 标题「STRIDER demo video」，上传者 Yuanzhuo Li，Unlisted，2026-09-20 |
| 同机构平台 | [Embodied Tien Kung 3.0 PR](https://www.prnewswire.com/news-releases/x-humanoid-introduces-embodied-tien-kung-3-0--a-more-open-and-practical-humanoid-robotics-platform-302688505.html) | X-Humanoid 通用 embodied 平台新闻稿 |
| 同机构开源 | [Open-X-Humanoid/XR-1](https://github.com/Open-X-Humanoid/XR-1) | 跨形态 VLA（ICML 2026 Oral），与 STRIDER 无直接代码关联 |

## 摘要级要点（来自标题与机构语境，**非论文结论**）

- **问题方向：** 人形机器人在 **三维空间** 中同时完成 ** locomotion + manipulation**，且需在 **多种步态** 间切换（含 stepping），而非单一行走模式。
- **方法线索（命名推断，待论文核实）：** **Hierarchical** 框架 — 低层多 gait / stepping 与高层 loco-manipulation 任务解耦或分层合成；**Stepping-Enabled** 暗示显式落脚/换步能力服务 upper-body 操作空间。
- **公开证据边界：** 仅有 demo 视频；**无量化 benchmark、无方法细节、无开源链接**；不宜作为技术选型或引用依据。

## 核心摘录（面向 wiki 编译）

### 1) 开源与复现（步骤 2.5）

| 组件 | 状态 |
|------|------|
| 演示视频 | 可观看（YouTube Unlisted） |
| 项目页 / 论文 PDF | **截至入库日未公开** |
| 训练 / 部署代码 | **截至入库日未公开** |

### 2) 与相近工作的阅读坐标

- 分层 loco-manip：[VisualMimic](../papers/visualmimic_arxiv_2509_20322.md)
- 机构线：[XR-1 仓库](https://github.com/Open-X-Humanoid/XR-1)

## 对 wiki 的映射

- 新建实体页：[paper-strider](../../wiki/entities/paper-strider.md)（**演示级** 条目，待正式论文后升级）
- 交叉：[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[whole-body-control](../../wiki/concepts/whole-body-control.md)

## 当前提炼状态

- [x] YouTube 元数据与机构归属核查
- [ ] 正式论文 / 项目页发布后再补方法细节与评测
