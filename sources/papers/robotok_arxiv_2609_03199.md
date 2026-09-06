# RoboTok: An Internet-Scale Data Engine for Human Demonstration Retrieval and Dexterous Manipulation Learning（arXiv:2609.03199）

> 来源归档（ingest）

- **标题：** RoboTok: An Internet-Scale Data Engine for Human Demonstration Retrieval and Dexterous Manipulation Learning
- **简称：** RoboTok
- **类型：** paper / dexterous-manipulation / data-engine / web-video / retrieval
- **arXiv：** <https://arxiv.org/abs/2609.03199>
- **项目页：** <https://rice-robotpi-lab.github.io/RoboTok/>
- **代码：** <https://github.com/Rice-RobotPI-Lab/RoboTok-Code> — 归档见 [`sources/repos/robotok.md`](../repos/robotok.md)
- **作者：** Howard Qian、Yiting Chen、Yunfei Xie、Kejia Ren、Podshara Chanrungmaneekul、Gaotian Wang、Bowen Wen、Chen Wei、Kaiyu Hang
- **机构：** Rice University；NVIDIA 等
- **入库日期：** 2026-09-06
- **索引来源：** [具身智能小站 9 篇资源汇总](../blogs/wechat_embodied_station_9_papers_resources_2026-09-06.md)
- **一句话说明：** 躯干相对 3D 手部轨迹嵌入空间索引互联网人类视频；DTW 监督编码器实现跨视角/遮挡检索，提升 VTDexManip 下游策略成功率。

## 开源状态（步骤 2.5，2026-09-06）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（检索 demo、基准表、仿真 rollout 视频） |
| GitHub | **已链出** [Rice-RobotPI-Lab/RoboTok-Code](https://github.com/Rice-RobotPI-Lab/RoboTok-Code) |
| 真机 | 项目页标注 **coming soon**（截至入库日） |

**结论：已开源**（检索/训练代码仓已发布）；真机结果待跟进。

## 核心摘录

### 摘录 1：方法

- 过滤互联网操作片段 → 3D 手关键点 → 躯干中心坐标系轨迹。
- DTW 对齐监督轻量编码器；余弦检索 + 持续索引新片段。
- 与外观/语义检索（Flow、HAND、STRAP）对比，in-domain mAP@20 **0.353** vs STRAP **0.007**。

**对 wiki 的映射：** [paper-robotok](../../wiki/entities/paper-robotok.md)

### 摘录 2：下游策略

- VTDexManip 6 任务：RoboTok 检索示范在 seen/unseen 多任务 **90%+** 成功率（easy）。
- Hard 设定（自由手运动、稀疏奖励）：Lever Sliding seen **79.3%** vs STRAP **8.4%**。

**对 wiki 的映射：** [paper-robotok](../../wiki/entities/paper-robotok.md)

## 当前提炼状态

- [x] 项目页 + GitHub 核查（2026-09-06）
- [x] wiki 映射：`wiki/entities/paper-robotok.md`
