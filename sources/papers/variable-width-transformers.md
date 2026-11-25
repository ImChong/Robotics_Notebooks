# Variable-Width Transformers｜可变宽度 Transformer

- **类型：** paper
- **作者：** Zhaofeng Wu、Oliver Sieberling、Shawn Tan、Rameswar Panda、Yury Polyanskiy、Yoon Kim
- **机构：** 麻省理工学院（MIT）、MIT-IBM 沃森人工智能实验室（MIT-IBM Watson AI Lab）
- **发布日期：** 2026-06-16（arXiv v1）
- **原文：** <https://arxiv.org/abs/2606.18246v1>
- **HTML：** <https://arxiv.org/html/2606.18246v1>
- **PDF：** <https://arxiv.org/pdf/2606.18246v1>
- **代码：** <https://github.com/ZhaofengWu/variable-width-transformers>
- **源码归档：** [官方代码与运行入口](../repos/zhaofengwu-variable-width-transformers.md)
- **入库日期：** 2026-11-25
- **核对范围：** arXiv v1 全文 §1–7、图 1–9、表 1–5、附录 A–C；重点核对逐层残差回填、参数匹配推导、对照/消融结果、效率外推与部署局限。
- **通读：** 已完成

## 原文要点

- §2 / 附录 A：按层变化的是解码器块的隐藏宽度，不是 token 序列长度。等参数约束主要匹配各层宽度平方和，并计入保持输入、输出嵌入宽度时的端点修正；窄层未计算的坐标可绕过该层，在后续宽层从最近包含它的状态回填。
- §3 / 表 1–3：在 DCLM 上从头训练 200M–2B 稠密模型和 3B 总参数 / 1B 激活 MoE。默认瓶颈约在深度的 75%，宽为基线的 30%。参数匹配的 2B 对照中，loss 由 2.751 降至 2.726，预训练 FLOPs 降 2.5%，平均层宽降 10.9%；MoE 的下游准确率有升有降。
- 图 4：取 2B 等宽模型的 loss 2.751 为目标，在作者的拟合曲线上，可变宽模型使用 77.8% 预训练 FLOPs、85.1% 平均层宽；不是同一模型直接测得 22% 的硬件加速。
- §4–5 / 附录 C：500M 消融支持残差坐标回填，2B 分析显示窄层附近有效表征利用更均匀；不规则内核形状、切片/复制开销和并行切分仍是实用限制。

## 对 wiki 的映射

- [Variable-Width Transformers 详情](../../wiki/entities/paper-variable-width-transformers.md)：方法、证据、复现与迁移边界。
