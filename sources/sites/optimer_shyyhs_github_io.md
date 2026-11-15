# OptiMer 官方项目主页（2026-11-15）

- **标题：** OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training / OptiMer：用最优分布向量合并替代持续预训练中的数据混合
- **类型：** site
- **项目：** https://shyyhs.github.io/optimer/
- **论文：** https://aclanthology.org/2026.emnlp-main.1090/
- **论文 PDF：** https://aclanthology.org/2026.emnlp-main.1090.pdf
- **代码：** https://github.com/nict-astrec-att/optimer
- **作者/机构：** Haiyue Song、Huy Hien Vu、Masao Utiyama；日本国立信息通信研究机构（NICT）
- **入库日期：** 2026-11-15
- **核查范围：** 打开项目主页，核对论文、Code 链接、方法、实验图与局限，并分别打开官方论文与源码仓库。
- **开源结论：** **部分开源**——有可运行的合并权重搜索脚本，未在仓库发现完整 CPT 训练流程或论文中的训练 checkpoint。

## 主页要点

- **正式出版：** EMNLP 2026 Oral，Outstanding Paper Award（以 ACL Anthology 论文记录交叉确认）。
- **核心方法：** 每种数据分布单独进行 CPT，计算相对于同一 PT 基座的 distribution vector，使用 TPE / Bayesian optimization 搜索向量合并系数，并额外合并 instruction-tuned vector。
- **主要结果：** 主页写搜索阶段相对 DataMix 快 15–35×、平均分超越基线 2.1–6.7 分，支持按任务重新优化已训练向量池。
- **作用边界：** 主页报告的是语言模型的 CPT / checkpoint 合并，并未给出机器人 VLA、WAM 或策略控制实验；对这些方向的迁移属于研究假设，非本文实验结论。
- **原始数据及模型：** 官方源码 README 提供输入 checkpoint 路径的示例，而非发布全部训练数据和 pretrained/CPT checkpoint。可运行部分为后验权重搜索和评测。

## 对 wiki 的映射

- [OptiMer 论文实体](../../wiki/entities/paper-optimer-optimal-distribution-vector-merging.md)
- [正式论文归档](../papers/optimer_arxiv_2603_28858.md)
- [源码审阅归档](../repos/optimer_nict_astrec_att.md)
