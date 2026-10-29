# AdaOcc: Adaptive 3D Occupancy Prediction for Embodied Tasks

- **类型：** paper
- **论文：** <https://arxiv.org/abs/2609.38864>；[HTML v2](https://arxiv.org/html/2609.38864v2)
- **作者：** Jinglong Wang、Yunjie Wang、Zhiyang Zhang、Jiawei He、Ye Yuan、Bo Qiu、Jing Zhang
- **机构：** 北京航空航天大学、北京智源人工智能研究院、河北工业大学、XYZ Embodied AI、上海科技大学、北京科技大学
- **版本：** v1 2026-09-30；v2 2026-10-10；项目页称已获 NeurIPS 2026 接收。
- **项目页：** <https://wangjl-nb.github.io/AdaOcc_web/>
- **代码：** <https://github.com/wangjl-nb/AdaOcc>；**权重：** <https://huggingface.co/wjldragon/AdaOcc>
- **入库日期：** 2026-10-29

## 摘要要点

1. RGB 与估计深度、深度相机或 LiDAR 几何输入通过双分支编码，几何统一成点集；支持不同视角数量。
2. 稀疏语义点经渐进 query 学习和多层解码预测占据，通过 query 数与中间解码层调整精度和预算；containment loss 约束点位于有效占据区域。
3. Occ-ScanNet 正式划分报告 IoU 65.29、11 类 mIoU 59.67；论文还展示 Go2 导航、ARX X5 操作及移动操作。小薄物体、遮挡和深度估计误差仍是局限。
4. 公开仓包含 OccScanNet-mini 训练评测代码、完整划分评测配置及权重；并非论文全部真实机器人演示均提供可复现入口。详见 [源码归档](../repos/adaocc.md)。

## 对 wiki 的映射

- [AdaOcc 实体](../../wiki/entities/paper-adaocc.md) — 方法、评测与复现边界。
- [Humanoid Occupancy](../../wiki/entities/paper-notebook-humanoid-occupancy-enabling-a-generalized-multim.md) — 多模态人形占据感知的相关工作。
