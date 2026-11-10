# Apple AMP1 官方项目页：On Agentic Manipulation Policies

> 官方原始站点与资源入口归档。

- **类型：** site / project-page
- **URL：** <https://apple-aiml-research.github.io/ml-amp/>
- **项目：** Apple AMP1 / AMP1+
- **论文：** <https://arxiv.org/abs/2611.09892>
- **代码：** <https://github.com/apple-aiml-research/ml-amp>
- **核验日期：** 2026-11-10
- **核对范围：** 官方 GitHub 仓 `docs/index.html` 中与项目网站同源的页面标记、Paper/Code 链接、方法段和基准卡片，及 README；网页直接抓取工具未成功返回完整动态页面。
- **通读：** 已完成

## 页面与公开状态

- 同源页面 `docs/index.html` 有 Paper → arXiv:2611.09892、Code → `apple-aiml-research/ml-amp` 的明确按钮。**官方评测代码已经公开**，不属于仅承诺“将开源”的状态。
- 方法简介明确：前沿模型**编写 Python**，调用 SDK 获取 RGB-D/标定/融合点云、可修正场景模型、末端定时轨迹与夹爪控制；多轮观察结果后继续决策。
- 项目站点基准卡片显示 RoboLab-120 约 **86%**（正文及精确实验以论文 **85.6%** 为准）、LIBERO-PRO **95.2%**、robosuite **100%**、BEHAVIOR-1K **96.0%**、RoboDojo 练习后 **84.1%**；这些不是同一评测协议。
- **误读警示：** 项目基准比较不足以证明“某模型没有上市”或“Claude/Gemini 在一切任务优于 GPT-6 Astra”。

## 与其他资料互指

- [论文来源归档](../papers/apple_amp1_arxiv_2611_09892.md)
- [代码来源归档](../repos/apple-ml-amp.md)
- [统一 wiki 论文节点](../../wiki/entities/paper-apple-amp1.md)
