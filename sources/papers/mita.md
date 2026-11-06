# Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights

> 论文来源归档（待完整核查）

- **论文：** <https://arxiv.org/abs/2602.01219v6>
- **作者：** Qishuai Wen、Zhiyuan Huang、Xianghan Meng、Wei He、Chun-Guang Li
- **机构：** 北京邮电大学
- **会议：** NeurIPS 2026（论文及作者官方仓库标注）
- **代码：** <https://github.com/QishuaiWen/MiTA>
- **版本：** arXiv v6，2026-09-27
- **核查状态：** 已核对摘要、方法、实验、部分附录及核心实现；全文和完整源码调用链尚待逐项复核。

## 内容摘录

- 自注意力可用输入生成的快权重表示为两层 MLP；原有注意力仍是实际计算。详见论文第 3.1 节。
- MiTA 用 landmark query 构建共享压缩专家，同时为每个 landmark 选取 top-k 键值对供查询路由复用。详见第 3.2 节。
- ImageNet-1K 中，MiTA-DeiT-T 为 71.1% / 1.1G FLOPs，加入深度卷积为 73.4%；基线 DeiT-T 为 72.2% / 1.2G。详见表 2。
- ADE20K 的 ViT-T 设置下，直接替换注意力后为 36.5 mIoU / 7G FLOPs；CBT 基线为 39.1 / 12G。详见表 4。
- LRA 五项任务合计训练时间由 10.6 小时降至 2.4 小时，论文在 RTX 4090 上测量，标准注意力使用融合内核。详见表 5。

## 对 wiki 的映射

- [MiTA 实体节点](../../wiki/entities/paper-mita.md)
