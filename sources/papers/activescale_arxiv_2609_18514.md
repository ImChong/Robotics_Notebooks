# ActiveScale: Scaling Active Perception for Robots across Model, Data, and Hardware

> 一手论文归档；仅保存书目与归纳，不转存全文。

- **类型：** paper
- **作者：** Shuai Zhou、Kaisheng Pang、Wenxuan Song、Wenjie Zhang、Xinhu Zheng、Haoang Li
- **机构：** 卡内基梅隆大学机器人研究所；香港科技大学（广州）
- **arXiv：** <https://arxiv.org/abs/2609.18514>
- **版本：** v2，2026-09-19；初版 2026-09-16
- **全文：** <https://arxiv.org/html/2609.18514v2>；<https://arxiv.org/pdf/2609.18514v2>
- **项目页：** <https://active-scale.github.io/>；[归档](../sites/activescale.md)
- **代码：** <https://github.com/ShuaiZhou302/ActiveScale>；[归档](../repos/activescale.md)
- **初次入库：** 2026-09-17；**重新核查：** 2026-10-10
- **一句话说明：** π0.5 联合学习相机与操作；几何监督历史 token、人机中训与 AMP 三臂平台配合。

## 核心摘录与映射

1. **§III-B：** Cobot-Magic 加相机臂，Quest 2 控视角/双臂/夹爪/底座；23D 位姿夹爪表示与另存底座速度。映射：[ActiveScale](../../wiki/entities/paper-activescale.md) 动作边界。
2. **§III-C：** 四帧间隔 16，9D 相机辅助目标与因果历史；推理不运行相机头。映射：同页机制/流程图。
3. **§III-D：** 人机 1:1 中训再任务适配，坐标与 mask 分来源；v2 列 EgoLive/EgoVerse，当前网页还列 EgoSuite，非相同口径。映射：同页数据与源码集成。
4. **§IV：** 五任务各 150 示范/20 rollout；完整方案 SR 30%→70%、TP 41.6%→78.4%。264.5 Hz 是吞吐、执行 30 Hz；移动展示是遥操作。映射：同页评测与[技术地图](../../wiki/overview/perception-action-transfer-9-papers-technology-map.md)。

## 核查边界

已读 v2 HTML 方法与实验，未重做训练。原 2026-09-17 判待发布；2026-10-10 官方页已有代码/权重/数据入口，见配套归档。任务数据目录不证明完整中训语料已释放。
