# FloMo（arXiv:2611.05062）

> 来源归档（paper）

- **标题：** FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.05062>
- **PDF：** <https://arxiv.org/pdf/2611.05062>
- **项目页：** <https://flomo-wam.github.io/>
- **代码：** <https://github.com/pairlab/flomo>
- **入库日期：** 2026-11-05
- **线索来源：** [人能稳稳提起的罐子，换一双机器手怎么提？](../blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)（具身智能小站，2026-11-05）
- **一句话说明：** FloMo 让模型预测场景中各点的三维位移和机器动作。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-05）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 学人手动作时，先预测桌上哪些点会动。作者在真实操作中用同一模型比较预测视频、预测运动和两者都预测，发现加入画面预测反而拉低了策略成功率。
- FloMo 让模型预测场景中各点的三维位移和机器动作。
- 指标：公众号转述：作者在真实操作中用同一模型比较预测视频、预测运动和两者都预测，发现加入画面预测反而拉低了策略成功率。

## 对 wiki 的映射

- [paper-flomo](../../wiki/entities/paper-flomo.md)
- [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md)
