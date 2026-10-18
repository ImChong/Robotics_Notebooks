# DR-WM 官方项目页

> 来源归档

- **类型：** site / project page
- **URL：** <https://developer.deeprobotics.cn/research/articles/dr-wm/index.html>
- **论文：** <https://arxiv.org/abs/2610.17630>
- **代码：** 暂未开放；页面标注 “Coming soon”
- **数据集：** 暂未开放；页面标注 “Coming soon”
- **作者 / 机构：** Ruo Chen、Jiahua Pang、Yingjie Niu、Hongyi Lin、Jun Tian、Chao Li、Yang Zhu；云深处科技、浙江大学、北京理工大学、麻省理工学院、清华大学、伊利诺伊大学厄巴纳—香槟分校
- **项目页发表日期：** 2026-10-13
- **入库日期：** 2026-10-18
- **一句话说明：** 官方页面展示 DR-WM 的地形物理建模、机器人动作条件未来预测、地形对比演示和论文结果；截至入库日，补充材料、代码和数据集均未开放。

## 项目页内容（归纳）

- **核心链路：** PhyMap 构建几何对齐的地形物理场，Genesis 依据动作预测机器人与相机运动，几何控制的视频模型生成未来观测，再更新下一轮地形重建。
- **演示：** 展示不同地形（原始地面、泥地、鹅卵石、雪地）以及多组动作条件的预测结果；并比较与 Cosmos 3、LingBot-World 等视频世界模型。
- **数据采集：** 页面展示 DEEP Robotics M20 与 Lite3 在多种地形上的同步 RGB、LiDAR、动作和机器人响应采集。
- **资源状态：** Paper 可访问；Supplementary、Code、Dataset 均注明 “Coming soon”。未发现公开代码仓库、模型权重或数据下载链接。

## 对 wiki 的映射

- 论文归档：[sources/papers/dr_wm_arxiv_2610_17630.md](../papers/dr_wm_arxiv_2610_17630.md)
- 沉淀：[wiki/entities/paper-dr-wm.md](../../wiki/entities/paper-dr-wm.md)
