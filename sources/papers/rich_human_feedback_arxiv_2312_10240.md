# Rich Human Feedback（arXiv:2312.10240）

> 来源归档（paper）

- **标题：** Rich Human Feedback for Text-to-Image Generation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2312.10240>
- **PDF：** <https://arxiv.org/pdf/2312.10240>
- **代码：** <https://github.com/google-research/google-research/tree/master/richhf_18k>
- **入库日期：** 2026-11-01
- **线索来源：** [四年十篇顶会Best Paper，谁在持续霸榜？](../blogs/wechat_shenlan_cv_best_papers_2023_2026_labs.md)（深蓝具身智能，2026-11-01）
- **一句话说明：** Rich Human Feedback 把人类对文生图的反馈从单一分数扩展为可定位的区域标注（不合理或与文本不符的区域、对应词），并训练多模态模型预测这些细粒度反馈。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：arXiv 摘要页链接 google-research/richhf_18k（数据集）；母仓库 google-research/google-research 经 git ls-remote 可达，子目录内容未逐文件核对。

## 核心摘录（公众号转述，未核验原文表格）

- 借鉴 LLM 的 RLHF，丰富 T2I 的反馈信号：区域标记 + 文本词标注 + 细粒度评分（公众号归属：Google Research 联合 UCSD 等）。
- 标注数据集 RichHF-18K，并训练多模态 Transformer 预测富反馈，用于挑选训练数据、微调生成模型或做区域修复（arXiv 摘要）。
- 指标：数据集名 RichHF-18K（名称暗示约 1.8 万条）；其余数字未转录。

## 对 wiki 的映射

- [paper-rich-human-feedback](../../wiki/entities/paper-rich-human-feedback.md)
- [从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标](../../wiki/overview/cv-best-papers-2023-2026-technology-map.md)
