# VINO 作者项目页

- **官方项目页：** <https://adriankruse.github.io/VINO/>
- **论文：** [VINO（arXiv:2611.07176）](../papers/vino_arxiv_2611_07176.md)
- **代码：** <https://github.com/AdrianKruse/VINO>（仅仓库占位 README；训练/推理代码待发布）
- **权重：** 项目页标注 “Weights (soon)”；**待发布**
- **入库/复核日期：** 2026-11-13
- **一句话说明：** Volt 骨干上重设计 3D self-distillation 与视角一致 2D→3D 原型监督。
- **源码开放程度：** **待发布**（仓库存在，但无可运行实现、无 checkpoint）。

## 项目页核验要点

- 3D-only 与 2D 蒸馏两张表各含 ScanNet200、ScanNet、ScanNet++、nuScenes、SemanticKITTI、Waymo；不可跨表直接声称“纯 3D 自监督”。
- Figure Method 的教师：干净全局视图与平衡原型；学生：强增强局部视图、移除几何位置的 masked global view；第二阶段才引入冻结 2D VFM 特征。
- 复现入口：目前仅论文描述，不能推导官方训练命令。

## 交叉链接

- [论文全文笔记](../papers/vino_arxiv_2611_07176.md)
- [代码状态归档](../repos/adriankruse-vino.md)
- [主 wiki 节点](../../wiki/entities/paper-vino.md)
- [Volt 项目页](./volt-project.md)
