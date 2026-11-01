# BrickGPT（arXiv:2505.05469）

> 来源归档（paper）

- **标题：** Generating Physically Stable and Buildable Brick Structures from Text
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2505.05469>
- **PDF：** <https://arxiv.org/pdf/2505.05469>
- **项目页：** <https://avalovelace1.github.io/BrickGPT/>
- **代码：** <https://github.com/AvaLovelace1/BrickGPT>
- **入库日期：** 2026-11-01
- **线索来源：** [四年十篇顶会Best Paper，谁在持续霸榜？](../blogs/wechat_shenlan_cv_best_papers_2023_2026_labs.md)（深蓝具身智能，2026-11-01）
- **一句话说明：** BrickGPT 从文本生成物理稳定、可互锁搭建的积木结构：自回归 LLM 逐块预测下一块，并在推理时用物理与装配约束做有效性检查和回滚。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：仓库 AvaLovelace1/BrickGPT 经 git ls-remote 可达；项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 把物理可行性放进生成过程，生成结果可由人或机器人实际搭建（公众号归属：CMU 计算机科学系）。
- 构建大规模物理稳定积木数据集与配套文本描述，next-token 预测下一块，推理时用物理定律与装配约束剪除不可行的 token（arXiv 摘要）。
- 指标：arXiv 摘要称生成结果稳定、多样、美观；具体数字未转录。

## 对 wiki 的映射

- [paper-brickgpt](../../wiki/entities/paper-brickgpt.md)
- [从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标](../../wiki/overview/cv-best-papers-2023-2026-technology-map.md)
