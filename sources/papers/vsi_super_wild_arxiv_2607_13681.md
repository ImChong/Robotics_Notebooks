# Towards Spatial Supersensing in the Wild（arXiv:2607.13681）

> 来源归档；论文全文尚未通读，待补充。

- **论文：** <https://arxiv.org/abs/2607.13681>（v1，2026-07-15；ECCV 2026）
- **作者：** Tianjun Gu, Tianyu Xin, Kuan Zhang, Bowen Yang, Kok-Chung Chua, Peize Li, Xinran Zhang, Yupeng Chen, Qiyue Zhao, Qinlei Xie, Jianhang Liu, Yucheng Lu, Yinan Han, Marco Pavone, Yiming Li
- **机构：** 清华大学、NVIDIA、斯坦福大学（项目页所列）
- **项目页：** <https://vsi-super-wild.github.io/>
- **代码：** <https://github.com/THUSI-Lab/VSI-Super-Wild>
- **数据：** <https://huggingface.co/datasets/THUSI-Lab/VSI-Super-Wild>（项目页提供；本次未能打开数据集页面）
- **已核对范围：** arXiv v1 摘要、项目页及代码仓库 README；PDF 超出检索工具大小限制，全文未通读，实验细节和局限待核对。

## 摘要与项目页要点

- VSI-Super-Wild 是针对自然场景长视频中空间超感知的评测基准，覆盖观察者、物体和环境三个维度。
- 数据包含 442 段视频、8 类场景和 6,980 个人工核验问答；最长视频超过 4 小时。
- 四项任务分别是运动方向回忆、地点时间排序、物体时间排序和连续物体计数。
- 作者归纳四类失败模式：空间坍缩、语义捷径、更新不足、实例混淆。

## 待完成

- 通读论文 v1 的方法、实验、局限与相关附录；核实结果和复现条件。
- 阅读评测入口与任务实现，建立统一 wiki 实体及项目流程图、源码运行时序图，补充关联页面。
- 运行仓库要求的完整检查；当前网络无法克隆仓库，未运行本地 preflight。

## 关联归档

- [项目主页](../sites/vsi_super_wild.md)
- [官方评测仓库](../repos/vsi_super_wild.md)
