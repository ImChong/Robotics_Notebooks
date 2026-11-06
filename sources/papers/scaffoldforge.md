# ScaffoldForge: Bioinspired generative design of 3D-printed bone scaffolds via conditional diffusion for large segmental bone defect reconstruction

> 来源归档草稿：尚未通读全文，不能作为完成的论文分析。

- **中文标题：** ScaffoldForge：利用条件扩散进行仿生 3D 打印骨支架生成设计，用于大段骨缺损重建
- **作者：** Yaohua Zhang、Hongyue Zhang、Huaqing Yang
- **期刊：** PLOS One，21(10)，e0361041，2026-10-28
- **原文（发表版本）：** <https://doi.org/10.1371/journal.pone.0361041>
- **入库日期：** 2026-11-02
- **一句话说明：** 使用三维条件扩散模型，根据骨小梁结构描述符生成骨支架设计。

## 当前核对范围

- **通读：** 未完成

PLOS 官方目录确认题名、英文作者和日期。当前环境未能打开期刊全文；以下仅据[论文摘要](https://www.citedrive.com/en/viewer/?doi=10.1371%2Fjournal.pone.0361041&route=cite)记录线索，尚未核对正文、实验细节与图表。

- 摘要称条件包括骨体积分数、骨小梁厚度、间距、数量、连通密度、结构模型指数、各向异性和渗透率；以经验性物理约束惩罚刚度、制造可行性和灌注相关问题。
- 摘要报告计算估计的抗压强度为 56.6 MPa；设计时间约 12 秒，对照拓扑优化约 8,400 秒。这些是论文报告的计算结果，并非实体支架实测。
- 摘要明确称所有结果均来自计算；物理、生物、制造及监管验证仍待完成。

## 待完成

- [ ] 获取并通读发表版本全文，核对模型、训练、评测、对照条件和局限。
- [ ] 核实论文数据可用性声明、官方代码、模型和数据链接；目前未找到可核实的官方代码仓库，不能据此判断是否开源。
- [ ] 核实后创建 wiki 项目实体及基于全文的 Mermaid 流程图；如有可运行官方源码，阅读入口并补充源码时序图。
- [ ] 运行仓库要求的 `make ci-preflight`，检查 GitHub Actions 后再考虑将 PR 标记为可审阅。
