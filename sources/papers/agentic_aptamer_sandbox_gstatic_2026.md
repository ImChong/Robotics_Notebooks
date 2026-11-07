# Aptamers as a sandbox for lab-in-the-loop agentic molecular discovery（Google Research，2026）

> 来源归档（paper）

- **类型：** paper（预印本）
- **标题：** Aptamers as a sandbox for lab-in-the-loop agentic molecular discovery
- **作者：** Zachary Cutts, John Coller, Rachel Chen, Hua Tang, Mathias Bellaiche, Ryder Easterlin, Dina Zverinski, Petar Sirkovic, Avinatan Hassidim, Yossi Matias, Pooja Rao（共同高级作者）, Annalisa Pawlosky（共同高级作者）；机构：Google Research / Stanford 遗传学系（H.T. 为 Google Research 访问学者）/ Google Cloud / Google DeepMind
- **PDF（本次通读的版本）：** <https://www.gstatic.com/science-ai/agentic_aptamer_sandbox.pdf>（27 页 A4，8,077,293 字节，首页日期戳 2026-10-12，PDF 创建时间 2026-10-16）
- **arXiv：** <https://arxiv.org/abs/2610.24806>（q-bio.BM，v1，2026-10-17 提交；摘要页标注 34 页、4 主图、1 主表、3 补充图、4 补充表）
- **Google Research 论文页：** <https://research.google/pubs/aptamers-as-a-sandbox-for-lab-in-the-loop-agentic-molecular-discovery/>（引用为 arXiv preprint arXiv:2610.24806 (2026)）
- **正式（同行评审）版本：** 截至入库日（2026-11-07）未查到期刊版本；论文 Data availability 写「完整去标识序列-活性与差异富集表将在同行评审发表时提供」。
- **版本对照：** 另下载 arXiv v1 PDF（34 页）做了抽查：标题、摘要、关键数字（如 42.1% 出现次数与 gstatic 版一致）、Table 1、Data/Code availability 与 Supplementary Table S4 均在；两版版式不同（27 页 vs 34 页），本次**以 gstatic 27 页版逐页通读**，arXiv 版仅抽查未逐页通读。
- **代码 / 数据：** 论文未给出专属代码仓库；未列 GitHub 链接。数据见下方「开放状态」。
- **入库日期：** 2026-11-07
- **线索来源：** Google Research 博客 *Grounding AI agents in biological reality: Building blocks for autonomous wet labs*（由同批其他节点负责，本库对应节点由其他同事创建，写作时尚未存在）
- **核对范围：** gstatic 版 PDF 全部 27 页：正文（摘要、引言、R.1–R.5 结果、讨论）、Methods 全部小节、Data/Code availability、生物安全声明、参考文献 1–38、补充图 S1–S3 图注、补充注释 1、补充表 S1–S4 全文。图内像素数据未读，数字以正文与表格文字为准；Fig.1–4 仅读图注。未运行任何代码（论文无代码）。
- **通读：** 已完成

## 摘要要点（人工转述）

用蛋白结合型单链 DNA 适配体（aptamer）做「实验室在环（lab-in-the-loop）」智能体分子发现的沙盒：每轮约 1000 个设计序列、约 10 天台面周期、约 3 美元/候选；在 α-凝血酶与肌红蛋白上跑三轮设计–测试–改进（共约 4,400 条测序构建体），在新设计候选上经磁珠校正的显著命中率逐轮上升；单克隆磁珠滴定与池化测序富集相关（Pearson r=0.90）。

## 核心摘录与数字（均读自本 PDF）

1. **为什么选适配体（R.1，Fig.1A）：** 序列本身即功能分子，免去基因合成、克隆、表达、纯化；蛋白/抗体流程 7 步、每轮 3–6 周；经典 SELEX 每轮 6 步、需 8–15 轮、起始库 10^13–10^15 分子；本流程 3 个台面步骤、约 10 天、约 3 美元/候选。
2. **轮次与命中率（Fig.2B，Supplementary Table S3）：** α-凝血酶新设计候选 0.0%（0/167）→ 26.0%（54/208）→ 42.1%（131/311），严格从头子集 R3 为 37.1%（106/286）；肌红蛋白 10.2%（20/196）→ 3.7%（9/244）→ 33.5%（184/549）。随机序列阴性对照各轮均 0.0%。
3. **零样本预测器基准（Fig.2D）：** 在 R3 肌红蛋白库（N=1,441，292 个命中）上测 12 个开源预测指标（9 个工具）；Boltz-2 iPTM ROC-AUC 0.499、ΔiPTM 0.391；仅 ViennaRNA 的 −ΔG_MFE 0.729 与 −ED 0.763 有信号。
4. **G-quadruplex 交叉反应与恢复（Fig.3A/B，Table 1）：** 肌红蛋白特异性 75.0%→19.1%→72.4%；R2 有 80.9%（72/89）结合物为交叉反应，其中 97% 为 G4；R3 把 72.3% 的池转向非 G4 骨架；H 型假结命中率 80.4%（37/46），扩展回文发夹 61.8%（47/76），凸起发夹 45.9%（67/146）；α-凝血酶特异性 0.0%→76.7%→98.5%。
5. **单克隆滴定（Fig.4，Supplementary Table S4）：** 8 个 α-凝血酶全长候选 T01–T08，6/8 浓度依赖结合；1 μM 终点荧光与 R3 富集 Pearson r=0.90（p=0.0021）、Spearman ρ=0.88；T01–T04 表观 KD 24.6–133 nM（初步，未达平台），严格从头 T05–T06 KD>1 μM。
6. **操作者复现（Methods「Controls and reproducibility」）：** 两名操作者独立筛 R3 池，log2FC 相关 r=0.84–0.85（全池）。
7. **智能体架构（Methods）：** R1 用 Google AI co-scientist 零样本构想；R2–R3 用自建四阶段管线（analyst → strategist → 并行 designer → 云 GPU 打分阶段），基于 Gemini Pro/Flash，模块化 Python 智能体编排框架。
8. **人机分工（Methods）：** 人类研究员执行湿实验（pulldown、PCR、测序）并把计数矩阵返还；R2–R3 的差异富集、背景回归、失败诊断等由智能体从原始计数独立完成；湿实验条件三轮保持不变。

## 开放状态（Data and code availability 章节）

- 数据：摘要统计、基准指标、代表序列、滴定数据、整理后的智能体推理痕迹见正文与补充表 S1–S4；完整去标识序列-活性/差异富集表**待同行评审发表后**提供；PDB 坐标 1PPB / 3RGK / 1N8Z 公开。
- 代码：论文称流程基于标准 LLM API 与多智能体编排模式，不是专有算法，Methods 与 Table 1 足以在任意带工具的智能体平台上复现；**未提供专属代码仓库**。所用工具为开源且许可宽松的包（ViennaRNA v2.6.4、PyDESeq2、Boltz-2、ESMFold2-NA、RhoDesign、RiboDiffusion、DNABERT-2、HyenaDNA、RNA-FM）；AI co-scientist 平台见其参考文献 [1]（Gottweis 等，Nature 655, 487–496, 2026）。
- 本次未打开独立项目页以外的仓库；论文无项目页 GitHub 链接。

## 生物安全与披露

三种靶蛋白（人 α-凝血酶、肌红蛋白、HER2 胞外域）均为非感染、非毒性人蛋白，BSL-1；寡核苷酸池由 IDT 合成并经 IGSC / HHS 框架的自动化序列生物安全筛查；未用人类受试者、临床样本或动物。作者为 Google 员工或访问学者，部分可能持有 Alphabet 股票（Competing interests）。

## 对 wiki 的映射

- [paper-agentic-aptamer-sandbox](../../wiki/entities/paper-agentic-aptamer-sandbox.md)
