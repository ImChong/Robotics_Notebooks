# One Policy, Many Embodiments: Unified Camera-Centric Action Geometry Pre-training for Heterogeneous Embodied Manipulation

> 来源归档（ingest）

- **标题：** One Policy, Many Embodiments: Unified Camera-Centric Action Geometry Pre-training for Heterogeneous Embodied Manipulation
- **短名：** UCAG-P
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.26058>
- **PDF：** <https://arxiv.org/pdf/2608.26058>
- **项目页：** <https://public-bots.github.io/UCAG-P>
- **代码：** <https://github.com/Public-BOTs/UCAG-P>
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 用相机可观测锚点运动统一手臂、人形与人手的异构动作空间，再翻译成本体控制。

## 开源状态（步骤 2.5）

- **待发布**：[`Public-BOTs/UCAG-P`](https://github.com/Public-BOTs/UCAG-P) 是 GitHub Pages / 图集仓；README 徽章写 **Code Release Soon**。截至入库日无可运行训练脚本。

## 核心摘录（面向 wiki 编译）

### 摘录 1：相机中心动作几何

- 共享目标不是本体专属控制量，而是图像与相机坐标系中的腕部 / 抓取中心锚点运动。
- 几何条件动作转换器结合目标本体运动学生成可执行控制。
- 训练数据：机器人与仿真 **4030 小时** + 人类示范 **2340 小时**。

**对 wiki 的映射：** [paper-ucag-p](../../wiki/entities/paper-ucag-p.md)、[VLA](../../wiki/methods/vla.md)

### 摘录 2：评测

- 单检查点、无基准特化微调：LIBERO **98.3%**，RoboTwin Easy/Hard **88.7% / 89.2%**，LIBERO-Plus 零样本 **82.0%**，RoboCasa GR-1 **62.0%**。

**对 wiki 的映射：** [libero-benchmark](../../wiki/entities/libero-benchmark.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-ucag-p.md`](../../wiki/entities/paper-ucag-p.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
