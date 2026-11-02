# VisProg（arXiv:2211.11559）

> 来源归档（paper）

- **标题：** Visual Programming: Compositional Visual Reasoning Without Training
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2211.11559>
- **PDF：** <https://arxiv.org/pdf/2211.11559>
- **项目页：** <https://prior.allenai.org/projects/visprog>
- **代码：** <https://github.com/allenai/visprog>
- **入库日期：** 2026-11-01
- **线索来源：** [四年十篇顶会Best Paper，谁在持续霸榜？](../blogs/wechat_shenlan_cv_best_papers_2023_2026_labs.md)（深蓝具身智能，2026-11-01）
- **一句话说明：** VisProg 用大语言模型的上下文学习把自然语言指令写成 Python 式模块化视觉程序，调用现成视觉模型逐步求解，无需任务专门训练。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：仓库 allenai/visprog 经 git ls-remote 可达；项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 把大模型推理与传统视觉模块组合，得到可解释的执行过程（公众号归属：AI2 PRIOR 团队）。
- LLM 生成模块化程序，程序每一行调用检测、分割、VQA 等现成模型或 Python 函数，中间结果传给后续步骤，并给出可解释的推理依据（arXiv 摘要）。
- 指标：arXiv 摘要称在 4 类任务上验证灵活性（含组合式视觉问答与图像编辑等）；具体数字未在本页转录。

## 对 wiki 的映射

- [paper-visprog](../../wiki/entities/paper-visprog.md)
- [从视觉编程到 4D 重建：CV 三大会 2023–2026 Best Paper 阅读坐标](../../wiki/overview/cv-best-papers-2023-2026-technology-map.md)
