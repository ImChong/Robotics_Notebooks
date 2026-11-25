# One Model to Rig Them All: Diverse Skeleton Rigging with UniRig | 论文归档

- **类型：** paper
- **作者：** Jia-Peng Zhang、Cheng-Feng Pu、Meng-Hao Guo、Yan-Pei Cao、Shi-Min Hu
- **机构：** 清华大学（Tsinghua University）、VAST
- **论文：** [arXiv:2504.12451v1](https://arxiv.org/abs/2504.12451v1) · [PDF](https://arxiv.org/pdf/2504.12451v1)
- **发表：** ACM TOG 44(4), SIGGRAPH 2025，DOI [10.1145/3730930](https://doi.org/10.1145/3730930)
- **项目页：** [作者主页](https://zjp-shadow.github.io/works/UniRig/)
- **代码：** [VAST-AI-Research/UniRig](https://github.com/VAST-AI-Research/UniRig)
- **核对范围：** arXiv v1 正文 §1–9 与补充材料：数据、骨架编码、骨点交互、实验协议、局限与附表。
- **通读：** 已完成
- **入库日期：** 2026-11-25
- **对应节点：** [UniRig](../../wiki/entities/paper-unirig-2504-12451.md)

- §4：Rig-XL 从 Objaverse-XL 筛选、分类与去重，含超过 1.4 万个已 rig 的资产；VRoid 补足动画人物、手部和辅助骨骼。
- §5–6：采样网格点云，形状编码器条件化自回归模型；树标记表示关节结构与分支。第二阶段通过点/骨交叉注意力和体素测地距离等几何特征预测蒙皮；论文还研究辅助骨骼属性。
- §7：VRoid、Mixamo 和变换/姿态增强的集合分开；表 3 的 VRoid J2J 为 0.0092，NBS 0.0205；表 4 逐顶点权重 L1 为 0.0028 对 0.02721。变形实验施加外部 Mixamo 动画，再比较顶点误差；不是动作生成评测。
- §8–9 与补充材料：人可校正自动结果；训练分布之外的抽象或极度风格化拓扑仍可能失败。论文实验权重与作者当前公开 checkpoint 的训练集不同。
