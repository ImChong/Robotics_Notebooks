# ControlNet（arXiv:2302.05543）

> 来源归档（paper）

- **标题：** Adding Conditional Control to Text-to-Image Diffusion Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2302.05543>
- **PDF：** <https://arxiv.org/pdf/2302.05543>
- **代码：** <https://github.com/lllyasviel/ControlNet>
- **入库日期：** 2026-11-01
- **线索来源：** [四年十篇顶会Best Paper，谁在持续霸榜？](../blogs/wechat_shenlan_cv_best_papers_2023_2026_labs.md)（深蓝具身智能，2026-11-01）
- **一句话说明：** ControlNet 锁定预训练文生图扩散模型，复用其编码层作骨干，并经零初始化卷积（zero convolution）学习边缘、深度、分割、人体姿态等空间控制条件。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：arXiv 摘要页链接的仓库 lllyasviel/ControlNet 经 git ls-remote 可达。

## 核心摘录（公众号转述，未核验原文表格）

- 让大型扩散模型获得精确的空间可控性，是可控生成的基础组件（公众号归属：斯坦福 Brown 媒体创新研究所）。
- 冻结生产级扩散模型，复制编码层作可训练分支，用零卷积逐步增加参数以避免有害噪声破坏微调（arXiv 摘要）。
- 指标：arXiv 摘要称在 Stable Diffusion 上用单条件或多条件、带或不带文本提示测试多种控制；具体数字未在本页转录。

## 对 wiki 的映射

- [paper-controlnet](../../wiki/entities/paper-controlnet.md)
- [从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标](../../wiki/overview/cv-best-papers-2023-2026-technology-map.md)
