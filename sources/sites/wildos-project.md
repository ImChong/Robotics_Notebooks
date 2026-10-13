# WildOS 项目页

> 来源归档

- **标题：** WildOS: Open-Vocabulary Object Search in the Wild
- **类型：** site（项目页）
- **URL：** <https://leggedrobotics.github.io/wildos/>
- **论文：** <https://arxiv.org/abs/2602.19308>（归档见 [wildos_arxiv_2602_19308.md](../papers/wildos_arxiv_2602_19308.md)）
- **代码：** <https://github.com/nasa-jpl/nebula2-wildos>（归档见 [nebula2-wildos.md](../repos/nebula2-wildos.md)）
- **数据集：** <https://huggingface.co/datasets/leggedrobotics/wildos>（归档见 [wildos-frontiers.md](../datasets/wildos-frontiers.md)）
- **机构：** NASA JPL、ETH Zurich RSL、Stanford University、UC Berkeley
- **核查日期：** 2026-10-13
- **一句话说明：** 以长程视觉语义评分增强几何导航图，在越野与城市环境执行开放词汇目标搜索。

## 页面核查结论

- 页面提供论文、arXiv、公开视频、代码、模型权重与数据入口，并注明已被 IEEE Transactions on Field Robotics（T-FR）接收。
- WildOS 用稀疏导航图维护空间记忆；ExploRFM 根据图像和文字查询预测视觉 frontier、可通行性和目标相似度；粒子滤波式模块融合多视角目标观测，估计深度视野外的粗略目标位置。
- 页面展示围栏通行、死胡同恢复、开放词汇搜索和城市导航实地视频；没有发布对应的完整数字结果表。
- **开放状态：已开源。** 官方代码采用 Apache-2.0，Hugging Face 发布视觉 frontier 标注集。部分上游主干权重需另行获取。

## 对 wiki 的映射

- [WildOS 详情节点](../../wiki/entities/paper-wildos-open-vocabulary-object-search.md)
