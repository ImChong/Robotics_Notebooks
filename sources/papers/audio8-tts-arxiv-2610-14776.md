# Training a Compact, Strong, and Realistic Text-to-Speech Model

> 来源归档（ingest）

- **论文：** <https://arxiv.org/abs/2610.14776v1>；[全文 PDF](https://arxiv.org/pdf/2610.14776v1)
- **作者：** Runyuan Cai、Yu Lin、Yiming Wang、Xiaodong Zeng；**机构：** AutoArk。
- **版本：** arXiv v1，2026-10-10。
- **方法：** 多语言监督微调后，先以词错误率和说话人相似度进行 GRPO，再加入与同句人声比较的真实性评价器与时长约束。
- **结果与限制：** 九语言 CV3 平均错误率从 SFT 的 5.41% 降至最终 4.86%，SIM 从 0.639 升至 0.661；论文指出比较基线的评估流程存在差异，评价器需要同句人声且得分是相对指标。
- **模型卡与论文差异：** 论文第 3 节描述 fast 分支并行预测余下码本，Realistic 模型卡则称 fast AR 条件依赖先前码本；未独立核实该差异。
- **关联实体：** [Audio8 TTS](../../wiki/entities/audio8-tts.md)
