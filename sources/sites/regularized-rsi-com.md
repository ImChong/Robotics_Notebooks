# RRSI 项目页（regularized-rsi.com）

> 来源归档

- **标题：** RRSI: Regularized Recursive Self-Improvement of Agent Harnesses
- **类型：** site / paper-project
- **链接：** <https://regularized-rsi.com/>
- **arXiv：** <https://arxiv.org/abs/2609.24972v2>
- **GitHub：** <https://github.com/google-research/rrsi>
- **入库日期：** 2026-10-03
- **机构：** Google Cloud AI Research、北卡罗来纳大学教堂山分校、斯坦福大学、圣路易斯华盛顿大学
- **一句话说明：** RRSI 论文项目页，包含方法概述、结果摘要、逐轮进化交互浏览器与 BibTeX；明确论文与代码仓库的对应关系。
- **沉淀到 wiki：** [wiki/entities/paper-rrsi-2609-24972.md](../../wiki/entities/paper-rrsi-2609-24972.md)

## 开源状态核查（2026-10-03）

| 资源 | 核查结果 |
|------|----------|
| **论文** | 页面提供 arXiv 论文入口 |
| **代码** | 页面 Code 链接指向 google-research/rrsi，仓库公开 |
| **演化浏览器** | 可交互浏览真实运行中候选、critic 反馈、selection 决策与 harness diff |
| **数据 / 权重** | 项目页未单列可下载数据集或权重；论文实现依赖外部模型服务与 benchmark 环境 |

## 阅读提示

- 观察完整轮次时，对照 proposal 里被允许的变更和 selection 的拒绝理由，尤其关注噪声门槛、额外 token 成本与组件剪枝。
- 项目页的 headline 数字是作者汇总；横向比较前先对齐每个任务域的 task 数、裁判、trial 窗口与 held-out 协议。
