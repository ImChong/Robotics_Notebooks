# Rhoda AI：网络视频预训练真机 Scaling 官方技术原文核对

- **类型：** 官方研究技术文章（非同行评审论文）
- **机构：** Rhoda AI Research（Key Contributor: Tongzhou Mu）
- **原文：** Does Scaling Web-Video Pre-training Help Real Robots Do Real Work?（2026-09-10），<https://www.rhoda.ai/research/scaling-web-video-pretraining>
- **官方索引：** <https://www.rhoda.ai/research>
- **核查日期：** 2026-11-21
- **核对范围：** 通读官方博客全文（The question、What we tested、Model setup、Measuring pre-training quality、A real industrial task、How we score a trial、Choosing which checkpoint to evaluate、How we ran the evaluation、Results 1–3、Limitations、Conclusion）及全部附录 Appendix A–F（评估协议与SOP、Beta后验与Fisher精确检验/Holm校正/logistic交互检验、预训练批大小与计算量设置、检查点选择、后训练曲线、DINO FD计算细节）；核对全文未提供代码/权重/数据仓库链接。本文为中文事实摘录而非原文转载。
- **通读：** 已完成

## 实验设计与核心方法（原文核对）

- **实验控制：** 在 [DVA](../../wiki/entities/rhoda-direct-video-action.md) 框架下，固定**逆动力学模型（IDM）**（在更广的多机器人数据集上训练一次并保持不变）、任务、后训练流程、评测 SOP 与推理频率；仅改变**因果视频模型**的参数规模（XS / S / M / L）与预训练算力（M 模型 0.08× / 0.18× / 0.37× / 1×）。
- **预训练数据与配方：** 通用网络视频（纯视频、无动作标注、未针对操作或第一视角筛选）。采用 Kaplan et al. 临界批大小策略：模型规模扫描对 XS–L 使用固定梯度步数预算直至验证损失趋平（因而规模扫描**未做算力/token 匹配**，模型均训练超过 compute-optimal 点以降低推理成本）；算力扫描取 M 模型同一预训练运行中的 4 个累积算力检查点。
- **预训练质量代理指标（DINO FD）：** 在预留网络视频片段上由模型从真值上下文单窗口预测未来帧，经视频 tokenizer 解码后用冻结 DINOv2 ViT-L/14（CLS token）提取特征，计算预测帧（400,000 帧）与真值帧（400,000 帧）分布间的 Fréchet 距离（EMA 权重，越低越好；相邻检查点波动 1–3%，不衡量时序连贯性）。
- **真机工业任务与指标：** 客户真实拆包任务（10 kg 轴承箱，14 步 / 6 个子任务，包装按纸板/塑料/垫纸三类分拣），以 **at-speed completion rate**（**100 秒内**无人工干预且全部目标达成）为指标。累计超过 **200 小时**真机评测，每策略 100–260 次试验（XS 表现明显落后，27 次后提前停止）。
- **检查点选择（Appendix D）：** 同一 M 模型后训练运行中不同检查点完成率可差 28 个百分点（48.0% vs 76.0%）；验证损失在最差检查点附近见底并随后上升，无法选出好策略；团队改用无需真机试验的内部代理指标极小值选点。

## 核心定量结果（公司自报，95% 置信区间）

- **Result 1（模型规模）：** XS 3.7%（1/27）→ S 65.0%（67/103）→ M 75.3%（195/259）→ L 84.7%（94/111）；L 尚未饱和。失败拆解显示更大模型主要缩短完成时间、减少重试图次（超时未完成率从 S 的 27.2% 降至 L 的 10.8%）。
- **Result 2（预训练算力 × 后训练数据）：** 全量数据下 M 模型 0.08× 57.8%（59/102）→ 0.18× 67.3%（68/101）→ 0.37× 74.5%（111/149）→ 1× 75.3%（195/259）；25% 数据时 1×（60.2%）与 0.18×（29.1%）差距拉大至 31 个百分点；12.5% 数据时 1× 25.5% vs 0.18× 11.8%；交互检验 $\chi^2(1)=3.2, p=0.07$。
- **Result 3（DINO FD 相关性）：** 7 个预训练检查点（DINO FD：L 9.37、M 1× 14.84、M 0.37× 19.31、M 0.18× 22.44、S 30.68、M 0.08× 35.54、XS 91.41）按 DINO FD 严格同序对应真机完成率；且质量相近的 S（30.68）与 M 0.08×（35.54）在真机上未出现统计显著分离（65.0% vs 57.8%）。

## 开放与身份核查

- **确认未开源**：截至 2026-11-21 原文及 [Rhoda AI 研究索引](rhoda-ai.md) 均无公开代码、模型权重、预训练/后训练数据集或独立 arXiv 预印本，亦未披露 XS–L 具体参数量与训练 token 数。

## wiki 映射

- [Rhoda：网络视频预训练的真机 scaling 研究](../../wiki/entities/rhoda-scaling-web-video-pretraining.md)
- [Rhoda DVA](../../wiki/entities/rhoda-direct-video-action.md)
- [Rhoda AI](../../wiki/entities/rhoda-ai.md)
