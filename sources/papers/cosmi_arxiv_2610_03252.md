# COSMI（arXiv:2610.03252）

> 来源归档（paper）

- **标题：** COSMI: COmpositional Synthesis of Multi-object Interactions
- **作者：** Daniel Eskandar、Ilya A. Petrov、Gerard Pons-Moll
- **机构：** University of Tübingen、Tübingen AI Center、Zuse School ELIZA、Max Planck Institute for Informatics
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.03252>
- **HTML：** <https://arxiv.org/html/2610.03252v1>
- **项目页：** <https://ptrvilya.github.io/cosmi/>
- **代码仓库：** <https://github.com/ptrvilya/cosmi>
- **提交日期：** 2026-10-02
- **入库日期：** 2026-10-22
- **一句话说明：** 从单物体交互片段组合出多物体人体动作数据，并训练一个支持 1–5 个对象的文本条件扩散 Transformer。

## 为什么值得保留

COSMI 把数据规模扩展与生成模型设计统一起来：先利用身体局部交互的可组合性扩增训练序列，再让模型显式建模对象与驱动身体部位之间的关系。它为人–物交互动作合成、数字人动画和具身数据生成提供了可追踪的研究入口。

## 核心摘录

- **数据组合：** 从 GRAB、BEHAVE、InterCap、OMOMO、ARCTIC 和 AMASS 等单物体/动作数据中，按接触规则截取原子交互片段；由 Qwen3 辅助语义标注，镜像补平左右侧偏，再以语义兼容性和几何接触/穿透检查筛选组合。
- **数据规模：** 论文报告 221,677 条序列、275.4 小时、59 名受试者、36 个对象（其中 7 个为可动对象），每段最多 5 个对象；摘要将序列数约写为 222k。
- **生成方法：** 单个文本条件扩散 Transformer 生成 4 秒人体与对象轨迹。共享对象槽位使模型可变对象数量；部位分配头判断对象由手腕、头、躯干或脚等部位驱动，决定使用相对关节坐标还是绝对对象位姿。
- **接触引导：** 在采样末段对预测干净样本做梯度修正，使手指—物体距离和支撑脚速度满足模型预测接触对应的界限，从而缩小抓握间隙并减少足滑。
- **评测结论：** 在留出未知椅子和五种未见交互组合的基准中，COSMI 的文本对齐与接触准确率优于论文比较的基线；引导消融显著减少足滑和手物间隙，但手指精度仍不及专门的抓取合成方法。

## 开源状态

截至 2026-10-22，论文和项目页称代码、模型与数据流水线将发布。项目页链接的公开 GitHub 仓库目前展示 README、许可证和资源目录；README 仍标注实现“即将发布”，因此当前**尚无可确认的可运行官方实现**。

## 对 wiki 的映射

- [COSMI 论文实体页](../../wiki/entities/paper-cosmi.md)
- [基于扩散模型的运动生成](../../wiki/methods/diffusion-motion-generation.md)
