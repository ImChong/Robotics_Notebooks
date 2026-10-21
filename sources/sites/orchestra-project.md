# Orchestra 项目页（youpengwen.github.io/Orchestra）

> 来源归档

- **标题：** Let the Mind Conceive, Let the Hand Achieve（Orchestra）
- **类型：** site（项目页）
- **URL：** <https://youpengwen.github.io/Orchestra>
- **论文：** <https://arxiv.org/abs/2610.18352>（归档 [orchestra_arxiv_2610_18352](../papers/orchestra_arxiv_2610_18352.md)）
- **代码：** <https://github.com/youpengwen/Orchestra>（归档 [orchestra](../repos/orchestra.md)）；页面将 Hugging Face 与 arXiv 标为 "Coming soon"（2026-10-17 核查）
- **入库日期：** 2026-10-17
- **一句话说明：** 任务智能体 + 改造 π0.5 + 进度评估器的长程操作框架演示页。

## 页面结构

| 区块 | 内容要点 |
|------|----------|
| 方法 | 任务智能体拆子任务并落地对象；π0.5 生成动作；评估器决定切换 |
| RMBench | 九任务 68.22%（Mem-0 42.00%，HarnessVLA 26.78%） |
| RMBench-Compose | ASL 7.35 对 2.986；去编排 / 记忆 / 落地分别 1.502 / 2.500 / 5.814 |
| 真机 | 四任务各 20 条指令，ASL 2.20 → 2.75，SSR 72.77% → 82.09% |
| 机构 | 页面未列 |

## 开源核查结论

代码仓库已链接；HF 资源标 coming soon，权重 / 数据未发布（截至 2026-10-17）。
