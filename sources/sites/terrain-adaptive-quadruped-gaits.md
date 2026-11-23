# Terrain-Adaptive Gaits 官方项目页

> 来源归档（site）；作者项目主页。

- **标题：** Learning Optimal Gaits and Styles for Terrain-Adaptive Quadruped Locomotion
- **类型：** site
- **项目页：** <https://guillaume10.github.io/terrain-gaits/>
- **论文：** <https://arxiv.org/abs/2611.25958>
- **作者：** Guillaume Bellegarda；Auke Ijspeert
- **入库日期：** 2026-11-23
- **核对范围：** 官方网页 Highlights / Method / Stage 1 / Which gait is best / Stage 2 / Zero-shot / Bipedal mode / Hardware / BibTeX；核对网页所载文字与视频说明，未将视频画面视为独立测量。
- **通读：** 已完成
- **代码：** 截至 2026-11-23 项目页未见可核实的公开源码下载 URL（页面“Code”文本未对应可查证仓库）；开放状态待后续复核。
- **数据/权重：** 页面未列可核实的下载入口。

## 页面给出的重要证据

- 使用 Stage 1 统一控制 11 种四足步态并依地形调节机身高度、离地高度、足端前后偏移；Stage 2 在冻结底层策略后按特定目标在线切换步态。
- 项目页展示 0.5 m 和 0.7 m 缝隙、0.21 m 箱体的真机案例；仿真展示超训练范围的 0.8 m 缝隙、0.55 m 墙体、0.4 m 台阶。
- 高程图在真机由深度相机端侧估计，Go1/Go2 使用的 Stage 1/Stage 2 网络不需修改。
- 重要边界：网页是项目演示说明；具体算法参数、评价口径与限制均应以 [arXiv 全文](https://arxiv.org/html/2611.25958v1) 为准。

## 对 wiki 的映射

- [独立论文节点](../../wiki/entities/paper-terrain-adaptive-quadruped-gaits.md)
- [论文归档](../papers/terrain-adaptive-quadruped-gaits_arxiv_2611_25958.md)
