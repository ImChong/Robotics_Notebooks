# Robust Motion In-Betweening（LaFAN1 配套论文）

> 来源归档（ingest · paper）

- **作者：** Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, Christopher Pal
- **发表：** ACM Transactions on Graphics, 39(4), 2020（ACM SIGGRAPH 2020）
- **论文：** [ACM DOI](https://doi.org/10.1145/3386569.3392480) · [arXiv:2102.04942](https://arxiv.org/abs/2102.04942)
- **数据 / 基准代码：** [Ubisoft La Forge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- **官方介绍：** [Ubisoft La Forge](https://montreal.ubisoft.com/en/automatic-in-betweening-for-faster-animation-authoring/)
- **入库日期：** 2026-10-23（2026-11-07 按 arXiv v1 全文通读复核）
- **版本：** arXiv:2102.04942v1（2021-02-09，对应 ACM TOG 39(4) Article 60 的作者版），pdftotext 提取全文
- **核对范围：** 通读摘要、§1 引言、§2 相关工作、§3 方法（数据格式、网络结构、ztta / ztarget 嵌入修饰、损失、对抗训练与超参）、§4 实验（Human3.6M 预测与行走子集过渡基准、LaFAN1 扩展实验、MotionBuilder 插件）、§5 讨论（数据消融等）、§6 局限与未来工作、§7 结论；参考文献仅浏览，附录 / 补充视频未读，未复现实验。
- **通读：** 已完成
- **一句话说明：** 研究以关键帧生成连贯动作过渡，并在 Human3.6M 与新建的 LaFAN1 动捕基准上评测。

## 核心摘录

1. 论文在 Human3.6M 子集与 LaFAN1 上评测动作关键帧之间的过渡生成及更长时域泛化。
2. Ubisoft 官方仓库明确关联该论文，公开 BVH 动作、评估脚本和插值基线。
3. 仓库材料证明数据和基准脚本开放；不能据此推断论文学习模型的完整训练源码也公开。
4. **对 Wiki 的映射：** [LaFAN1 数据集实体](../../wiki/entities/lafan1-dataset.md)

## 参考来源

- DOI：<https://doi.org/10.1145/3386569.3392480>
- Ubisoft La Forge：<https://montreal.ubisoft.com/en/automatic-in-betweening-for-faster-animation-authoring/>
- 数据与评估代码：<https://github.com/ubisoft/ubisoft-laforge-animation-dataset>

## 2026-11-07 通读复核补充（出自论文原文）

- **LaFAN1 定义（§4.3）：** 496,672 帧 @30 Hz，生产级动捕棚采集，5 名被试，Subject 5 做测试集；测试窗口每 40 帧一个，共 2232 个；训练统计在 50 帧窗口、步长 20 上计算；论文模型在 Subject 1–4 上训练 35 万次迭代。骨架 $j=22$ 关节，局部四元数 + 3 维全局根速度，另由脚趾 / 脚速度提取 4 维二值接触向量。
- **方法：** 在 ERD 式 LSTM 预测器上加两个潜变量加性修饰——time-to-arrival 嵌入（ztta，解决变长过渡）与调度式目标噪声（ztarget，使生成可采样），再配两个多时间尺度的 LSGAN 判别器（滑窗 10 帧与 2 帧瞬时运动）；以 10 帧过去上下文 + 单个目标关键帧为输入；损失权重 $L_{quat},L_{root}=1.0$、$L_{pos}=0.5$、$L_{gen},L_{contacts}=0.1$。
- **LaFAN1 基线（表 3，过渡 5/15/30/45 帧）：** L2Q 插值 0.22/0.62/0.98/1.25 → TGcomplete 0.17/0.42/0.69/0.94（NPSS 与 L2P 数值在 pdftotext 抽取中列序错位，未引用）。论文指出在 LaFAN1 上即使很短的过渡也优于插值，而 Human3.6M 行走子集上不然。
- **数据消融（§5.2）：** 去掉舞蹈序列（约 10% 数据）与只保留舞蹈序列（21.5 分钟）各训一个模型：只用舞蹈的模型与较大消融集效果相近，但要让过渡保持舞蹈风格仍需完整数据——「大量通用数据与少量专门数据结合最关键」。
- **局限（§6）：** 位置偏移需要 FK，在网络内逐步做 FK 使反向传播过慢，故只用根偏移与旋转偏移；条件不现实或超出训练分布时生成变差；调度目标噪声只能在一定程度上改变到达方式，无法控制变化；风格控制超出范围。
- **注意：** 论文是动画领域的过渡生成工作，**不涉及机器人**；机器人侧对 LaFAN1 的用法（重定向、AMP 参考）来自后续论文，不在本文范围。
