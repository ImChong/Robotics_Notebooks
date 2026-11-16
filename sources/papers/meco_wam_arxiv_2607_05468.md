# Learning 4D Geometric Priors for Inference-Efficient World Action Models

- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2607.05468>
- **全文版本：** <https://arxiv.org/pdf/2607.05468v1>
- **核对日期：** 2026-11-13
- **核对范围：** arXiv v1 PDF，全文 Abstract–Conclusion（8 页正文），图 1–6、表 1–4；参考文献浏览，无单独附录。图表以文字层核对，未观看演示视频，未运行训练或真机实验。
- **通读：** 已完成
- **项目页：** <https://meco-wam.github.io/>

## 通读核对

- 方法：冻结 VGGT-1B 生成当前/未来几何 targets；约 0.45B 4D expert，只对选定 keyframes 预测算关系损失。未来几何不能作为动作捷径；当前 g0 的 read edge 概率在前半训练从 1 线性衰减至 0。
- 动作感知蒸馏：video/action 相似度产生区域权重，与均匀先验混合以防塌缩；同时匹配帧内几何距离关系及相邻 keyframe 的关系变化。
- 实现：H=32，9 帧视频、4×动作/视频比例，Wan2.2-TI2V-5B，64×H20 训练；单 RTX 5090、10 步推理；部署移除 VGGT / 4D expert / alignment。
- 表 1–2：LIBERO 97.6→98.2；RoboTwin 平均 91.83→92.62（clean 93.26、randomized 91.98）；不是所有基准绝对最高。
- 表 3：真机堆块 SR 60→60 持平，仅修正次数/时间改善；尺寸排序 SR 60→70。表 4：孤立 4D expert 只到 91.87，完整组合 92.62。
- 项目页本次 HTTP 404，不能核实源码开放；论文未给可运行仓库。不据此臆造源码时序图。

## 对 wiki 的映射

- [Learning 4D Geometric Priors for Inference-Efficient World Action Models](../../wiki/entities/paper-meco-wam-4d-geometry-cotraining.md)
