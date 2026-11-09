# Aligning Policy Encoders with a Shape Teacher（arXiv:2611.08275）

> 来源归档（paper）

- **标题：** Aligning Policy Encoders with a Shape Teacher（项目页标题写作 Aligning Policies with a Shape Teacher）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.08275>
- **PDF：** <https://arxiv.org/pdf/2611.08275v1>（v1，2026-11-05，cs.RO）
- **项目页：** <https://miepjemadelon.github.io/aligningpolicieswithashapeteacher/>
- **代码：** 未核实到（入库日 2026-11-09 项目页仍是未填充的学术项目页模板，GitHub 链接为占位符 `YOUR REPO HERE`；论文未承诺开源）
- **入库日期：** 2026-11-09
- **作者与机构（以论文首页为准）：** Madelon Bernardy、Mohammad Mahdi Derakhshani、Riccardo Valperga（阿姆斯特丹大学 University of Amsterdam）；David M Knigge、Artem Moskalev（New Theory AI；首页作者块中 Knigge 列在 New Theory AI 下，Valperga 列在 UvA 下，Moskalev 列在 New Theory AI 下）
- **会议：** NeurIPS 2026 Workshop: Second Embodied Spatial Reasoning Workshop（首页脚注）
- **一句话说明：** 训练时用冻结的 Point-MAE 三维形状特征，按像素反投影对应关系对齐 2D 扩散策略图像编码器的 patch 特征；部署时只用图像，无额外推理开销。
- **核对范围：** 读了 v1 PDF 全文（摘要、第 1 引言、第 2 方法含式 1 至 5、第 3 实验含表 1 与表 2 及图 1 至 5、第 4 结论与局限、参考文献、附录 A 模型架构、B 形状分组实验细节、C 对齐权重 λ 迭代、D 额外目标消融含表 4、E 数据集）；图表按文字与坐标轴读取，图 3、图 4、图 7、图 8 的柱高与曲线数值只读了图内标注的倍数与百分比，未逐点读曲线；未读论文引用的其他论文原文，未运行任何代码（项目页无代码）。
- **通读：** 已完成

## 核心摘录

1. **方法（第 2 节）：** 基线为 Chi et al. 的扩散策略；条件为 t-1 与 t 两帧观测，预测 10 步动作轨迹并执行 8 步；观测编码器 ViT/S-14（去掉 CLS，保留 patch token），语言由冻结 CLIP 编码，解码器为 12 层交叉注意力解码器加 2 层条件编码器，交叉注意力输出投影零初始化。
2. **形状教师对齐（第 2 节，式 1 至 5）：** 对每个物体用 mask 与深度图构造点云，重采样到 1024 点，每 M = 32 点一组，由冻结 Point-MAE 给每组一个特征；点反投影到像素后，按点落入 patch 的比例 φ(a,p) 加权聚合出每个 patch 的目标 t_p^j；对齐头结构同 Li et al. 2025（Spatial Forcing），损失为覆盖加权余弦距离，总损失 L = L_bc + λ L_align。
3. **设置（第 3 节）：** 仅 LIBERO-Object 与 LIBERO-Spatial；预训练策略先在 LIBERO-90 上训练再微调；另造 Object-Displacement、Spatial-Displacement（目标放在原位置周围两圈 ring，附录 E）并用 LIBERO-Occ 测遮挡零样本鲁棒性；默认 Point-MAE 教师、λ = 3，三个随机种子。
4. **表征（第 3.1 节，表 1）：** LIBERO-Object 上，预训练起点下「最相似 patch 落在形状相似的另一物体」从 8.2% 升到 35.1%，「无关区域」从 81.6% 降到 56.0%；从零训练下相似形状 10.7% 到 36.4%，同一物体 17.8% 到 1.3%（附录 B：1000 张图平均，按表 3 的形状分组）。
5. **收敛（第 3.2 节，图 3，表 2）：** 达到基线同样成功率所需训练量，预训练起点 Object 4.7 倍、Spatial 5.6 倍；从零训练 Object 2.3 倍、Spatial 3.9 倍。第 100 轮成功率（表 2）：Object 预训练 0.85 到 0.90，Object 从零 0.10 到 0.53，Spatial 预训练 0.71 到 0.82，Spatial 从零 0.20 到 0.62。
6. **鲁棒性（第 3.3 节，图 4）：** 预训练策略零样本；遮挡下 Object 与基线降幅相近、Spatial 降幅更小；位移下 Object 降幅更小、Spatial 差异较小；作者自述「并非对所有分布偏移一致提升」。
7. **目标消融（第 3.4 节，表 2，表 4，图 5，图 8）：** 换成 DINOv2（λ = 1）：第 100 轮成功率 0.91、0.75、0.85、0.35（顺序同上），加速 8.6 倍、不少于 11.1 倍、8.9 倍、3.7 倍；DINOv2 patch 相似性偏同物体，Point-MAE 偏跨相似形状物体与对应部位。
8. **λ 扫描（附录 C，图 7）：** λ 存在甜点，超过后可能变差，从零训练的 Object 尤其明显；作者自述需要更多配置与种子才能下结论。
9. **作者自述局限（第 4 节）：** 单一策略架构、仅 LIBERO、单一 Point-MAE 教师；鲁棒性不一致；真机与其他架构待验证。

## 对 wiki 的映射

- [paper-shape-teacher-alignment](../../wiki/entities/paper-shape-teacher-alignment.md)
