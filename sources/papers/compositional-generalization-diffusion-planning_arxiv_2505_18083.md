# What Do You Need for Compositional Generalization in Diffusion Planning?（arXiv:2505.18083）

> 来源归档（paper）

- **标题：** What Do You Need for Compositional Generalization in Diffusion Planning?（v1/v2 题为 *What Do You Need for Diverse Trajectory Stitching / Composition in Diffusion Planning?*，v3 起改为现名）
- **类型：** paper
- **作者：** Quentin Clark, Florian Shkurti（多伦多大学计算机系；致谢 Vector Institute 提供算力）
- **arXiv：** <https://arxiv.org/abs/2505.18083>
- **PDF：** <https://arxiv.org/pdf/2505.18083v3>（v3，2026-02-09；v1 2025-05-23）
- **代码：** 论文正文与附录未给出代码或项目页链接（已全文检索 github / project page / code）
- **入库日期：** 2026-11-12
- **一句话说明：** 在生成式行为克隆（BC）训练的扩散规划器上，把「拼接 / 组合泛化」归因于局部感受野、平移等变性与推理选择，并据此提出纯卷积的 Eq-Net。
- **核对范围：** 通读 arXiv v3 PDF 正文（Sec. I–V 含引言、方法、实验、讨论、局限）与附录 VI-A 至 VI-I（环境细节、超参数表 II–V、Eq-Net 与核大小实验 Table VI、其他因素 Table VII–IX、梯度图、对其他规划器的解释）；同时读了 v1 PDF 的正文与附录用于对照版本差异。图 2、5、6、7、8、10 的柱 / 曲线具体数值只在图中，文本抽取读不到，本归档不转录这些数值；未读参考文献所引的其他论文。
- **通读：** 已完成

## 开源状态

- 正文与附录均未给出代码仓库或项目页（步骤 2.5：无项目页可核查，视为未公开；以后若作者补链接再更新）。

## 核心摘录（均出自 v3，括号为章节 / 表号）

1. **问题（Sec. I, III-A, Fig. 2）：** 标准扩散规划器的 U-Net 与 DiT 骨干在作者设计的 6 个测试环境中给出高比例「记忆」轨迹（记忆 = 新轨迹只由同一条训练轨迹中的子技能构成）。具体百分比只在图 2。
2. **两个性质（Sec. III-B, Def. 1/2）：** 局部感受野（某步去噪预测条件独立于远处步）与平移等变性（模型忽略状态在序列中的绝对位置）。论文把它们借自图像扩散的机理研究（Kamb & Ganguli；Niedoba et al.）。
3. **三类干预（Sec. III-C, Fig. 3）：** 推理时重规划、数据增广（随机左移轨迹并用末状态补齐）、架构修改（Eq-Net：无池化 / 下采样的残差 1D 卷积，小核）。
4. **Eq-Net（附录 VI-E）：** 步长 1 卷积、无池化，核从 3 起每 KER 层增大 2；KER 取 10 在 1000 次生成上完成率 0.995、新颖率 0.802（Table VI，KER=6 / 16 时分别为 0.5056 / 0.9701 新颖率，完成率 0.991 / 0.965）。
5. **实验环境（Sec. III-D, Table V）：** Maze（Numpy 点机器人）、Didactic（真机 Franka）、Block-Stack 与 Well-Plate Sim（RoboSuite）、Well-Plate Real（真机 Franka）、Lights（改自 OGBench）。数据集 50 至 1000 条轨迹。
6. **架构对比（Sec. IV-A / IV-B, Fig. 5/6）：** 文字结论是 Eq-Net 在所有测试环境新颖率最高；仅局部（Eq-Net+PosEnc）优于仅等变（U-Net+Aug），二者结合最佳，局部性边际效应更大。
7. **与分块方法比较（Sec. IV-C, Fig. 7）：** Diffusion Policy、CompDiffuser 在操作任务上与 Eq-Net 表现相近；CompDiffuser 在导航环境因重叠点轨迹质量差而吃亏。
8. **数据规模（Sec. IV-D, Fig. 8–10, 附录 Fig. 13）：** 所有架构数据足够时都能组合，Eq-Net 所需数据最少；数据变多感受野也变小（用 50% 梯度幅值窗口的积分度量）。附录称完成率在各模型开始泛化的阈值附近出现下降。
9. **可引导性（Sec. IV-E, Table I）：** Maze-GC 上完成率 U-Net 0、U-Net/Aug 0.44±0.0、Eq-Net/PosEnc 13.39±5.0、Eq-Net 83.8±6.4（%）；唯一路径数 0 / 0.09 / 11.16 / 116.8。Didactic（真机）完成率 U-Net 0、DiT 0、Eq-Net 77（%）。正文写「83.3%、约 7.5 倍」，与表中 83.8 与 83.8/13.39≈6.3 不一致。
10. **局限（Sec. V-B）：** 环境为专门设计的需拼接任务；引导仅用修复（inpainting）；对语言条件或多模态目标不适用。附录 VI-E2 另承认 Eq-Net 不是有竞争力的通用架构，只是概念验证。
11. **算力（附录 VI-C4）：** 约 168 GPU 天（L40s 等价）。

## 对 wiki 的映射

- [paper-compositional-generalization-diffusion-planning](../../wiki/entities/paper-compositional-generalization-diffusion-planning.md)
