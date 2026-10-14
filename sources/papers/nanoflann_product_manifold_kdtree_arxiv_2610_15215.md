# nanoflann（arXiv:2610.15215）

> 来源归档（paper）

- **标题：** nanoflann: A Header-Only KD-Tree Library for Exact Nearest-Neighbor Search on Compile-Time Product Manifolds
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.15215>
- **PDF：** <https://arxiv.org/pdf/2610.15215>
- **代码：** <https://github.com/jlblancoc/nanoflann>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 把旋转、位姿等乘积流形的距离与剪枝规则编译期纳入精确 KD-tree 近邻搜索，避免对原始坐标直接用欧氏距离。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-14）：长期维护的 header-only C++ 库；乘积流形接口需 C++17。

## 核心摘录（公众号转述，未核验原文表格）

- 机器人状态空间的近邻搜索若忽略几何，采样规划与定位会得到错误邻居。
- 编译期组合各因子流形的度量与包围盒剪枝；另提供动态索引。
- 指标：库论文；性能对比以原文与仓库 benchmark 为准。

## 对 wiki 的映射

- [paper-nanoflann-product-manifold-kdtree](../../wiki/entities/paper-nanoflann-product-manifold-kdtree.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
