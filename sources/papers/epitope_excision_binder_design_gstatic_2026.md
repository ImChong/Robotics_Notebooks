# Agent-guided epitope excision enables economical de novo binder design

- **类型：** paper
- **作者：** Jun W. Kim, Artiom Myaskovsky, Oscar Donosa, Lawrence Chan, Jamie Rogers, Avinatan Hassidim, Yossi Matias, Annalisa Pawlosky（Kim 与 Pawlosky 为通讯作者，共同牵头）
- **机构：** Google Research；Google DeepMind；Google Cloud
- **论文 PDF：** https://www.gstatic.com/science-ai/epitope_excision.pdf
- **版本：** PDF 页眉日期 2026-10-23，共 18 页（正文 15 页 + Supporting Information 3 页），PDF 元数据创建时间 2026-10-23
- **arXiv 预印本：** https://arxiv.org/abs/2610.33195 （v1，2026-10-23，q-bio.BM；arXiv 注释称 24 页、6 主图、1 主表、5 补充图、1 补充表）。arXiv 摘要与 gstatic PDF 摘要逐字一致；两者页数不同（18 vs 24），本次只通读了 gstatic PDF，未逐页比对 arXiv 版排版差异。
- **正式发表版：** 截至 2026-11-07 用 arXiv API、bioRxiv API 与 Crossref 标题检索，未发现期刊或 bioRxiv 版本（检索无命中不等于不存在）。
- **核对范围：** gstatic PDF 全部 18 页：摘要、Introduction、Results（含图 1–6、表 1）、Discussion、Methods、致谢与利益声明、作者贡献、参考文献 1–17、Supporting Information（Supporting Fig. 1–5、Supporting Table 1）均已读；PDF 中没有 Code / Data availability 小节，也没有单独的生物安全小节。未读：arXiv 24 页版本全文、被引文献、图中未印出的数值。
- **通读：** 已完成

## 核心摘录

1. 问题：de novo 结合蛋白设计的算力瓶颈在 AlphaFold3（AF3）重折叠；AF3 按 256 token 分桶取整计时（Fig. 1c：256/512/768/1024 token 对应 6.9/17.5/35.0/59.3 s）。
2. 表位切除（epitope excision）：只保留结合体接触残基（5 Å 接触壳）加可调 halo 所在的连续序列窗口，用该裁剪口袋代替整条靶标折叠与设计（Methods「Target excision」）。
3. 15 靶标基准（Overath 等元分析的公开释出数据；3,666 条设计可定义连续窗口）：halo 11 Å 时界面 Cα RMSD 中位 0.37 Å，88.2% 低于 1 Å，93.7% 低于 2 Å（Results）。
4. 排序：ipSAEmax 的平均 AP 从全靶标 0.47 降到切除 0.40，ROC-AUC 0.78 到 0.73；关闭靶标 MSA 后切除 AP 回到 0.454；SARS-CoV-2 RBD 为唯一关 MSA 后变差的靶标（0.49 到 0.24）。
5. 裁剪下限：保留靶标残基数小于约 85 时通过 3 Å 门限仅 32%（n=41），85–200 为 97%（n=553）（Fig. 3d,e）。
6. FAP（PDB 1Z68 链 B，39–757，719 残基）：单个表位簇、85 残基裁剪、A100 上 RFdiffusion / ProteinMPNN / AF3 单位时间分别降 89% / 66% / 88%（Supporting Fig. 5a）。
7. 自主性阶梯（Table 1）：同一命中定义（ipSAEmax > 0.45 且界面 RMSD < 3.0 Å）与 132.87 GPU 小时预算上限；Arm A（PI 单独）5.06%、132.87 GPU-h、12.6 hits/GPU-h；Arm B（人机混合 HITL）9.33%、77.18 GPU-h、23.2；Arm C（智能体单独 HOOTL）6.18%、124.43 GPU-h、14.5；B-variant 32.62%（ipSAEmax 口径），属参数消融对照。控制器为 Gemini 3.1 Pro，每轮选一个类型化动作。
8. 湿实验：酵母表面展示，5,000 条寡核苷酸库（每臂 1,250 条设计，并加 2–3 个核苷酸随机突变），300/150/75 nM 三轮 FACS，富集到 327 条精确匹配设计与 5,034 条相关序列；四个代表克隆表观 KD 3.0–9.1 nM（Fig. 6）。
9. 局限（Discussion）：单一大酶靶标 FAP、单一骨架类别（三螺旋束）、单一 LLM 控制器；单序列折叠不适用所有靶标；可溶表达结合体表征、FAP 结构域特异性与催化抑制实验留作未来工作。
10. 经济性只以 GPU 时间 / token 桶 / GPU 小时计量，论文没有给出美元成本；论文也没有提到液体处理或自动化实验室，酵母展示由作者团队「in-house and with DuneX Biosciences」完成。

## 论文内部不一致之处（核对记录）

- 数据量口径：Results 称多引擎共识用 3,760 条设计，Methods 称公开释出 3,766 条、保留 3,760 条、切除保真分析用 3,666 条（排除 94 条 pMHC）。三者对应不同过滤阶段，不是笔误。
- Fig. 5d 图注称 B-variant「与 Arm C 相同的混合自主度」，而 Results 与 Table 1 脚注称其与 Arm B 同一自主度、仅改 MSA 与 ipSAEmin；本库按 Results / Table 1 记录。
- Results 称 Arm C 的策略「与人类研究者为 Arm C 设计的 steric-blockade 思路相近」，上下文应指 PI 设计的催化口遮挡策略（Arm A/B 所用），原文措辞为「Arm C」。
- 摘要说 FAP 的 RFdiffusion 与 AF3 GPU 时间降约 88%，正文 Supporting Fig. 5a 的 RFdiffusion 为 89%、AF3 为 88%。
- 摘要「roughly substantially increased the apparent yield」措辞不通顺，本库只引正文的 AP 数字。

## 对 wiki 的映射

- [Epitope Excision 论文实体](../../wiki/entities/paper-epitope-excision-binder-design.md)
