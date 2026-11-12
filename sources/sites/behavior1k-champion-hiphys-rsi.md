# HiPhys-RSI｜BEHAVIOR 2026 冠军项目主页

- **类型：** 官方项目站点 / 冠军方案说明
- **网址：** https://behavior1k-champion.com/
- **技术报告：** https://behavior1k-champion.com/assets/HiPhys-RSI_tech_report.pdf
- **团队：** The North Star；华为诺亚方舟实验室具身智能团队
- **竞赛规则：** https://behavior.stanford.edu/challenge/evaluation.html
- **组织方榜单：** https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard
- **代码：** 截至 2026-11-12，官网未列出冠军方案的可运行 GitHub 仓库或安装、部署、推理入口。
- **数据与权重：** 官网未发布 HiPhys-RSI 专有可下载权重或训练数据入口；报告说明 coding agent / VLM 冻结，下游 VLA 可单独后训练。
- **入库日期：** 2026-11-12
- **核对范围：** 官网头部、Abstract、Method 四阶段、跨任务复用、policy cards、Results、六个 Case Studies、Evaluation Integrity、BibTeX、Supporting Technical Papers 与页脚；与报告及竞赛规则交叉核对。
- **通读：** 已完成

## 要点

- 100 项家务任务隐藏测试：Q-score 0.683、完整成功率 48.2%，排名第一。
- 四阶段：Experience the task → Design the pipeline → Analyze rollouts → Refine iteratively。
- 设计阶段由冻结模型生成/修正可执行结构；线上由代码自主管理模块与观测反馈。Policy cards 负责选择既有 Pipeline 或 VLA。
- 记录能力边界与失败案例；主动披露局部评分漏洞，区分有效新策略和钻评价空子。

## 可复现性与误读

官方页面提供演示与 PDF，但并未发布冠军 Pipeline 具体源码。StanfordVL/BEHAVIOR-1K 是**赛事环境**的独立开源项目，不能冒充冠军解法的官方实现。

## 对 wiki 的映射

- [HiPhys-RSI 冠军方案实体](../../wiki/entities/paper-hiphys-rsi.md)
- [技术报告归档](../papers/hiphys-rsi-scaling-by-structure-2026.md)
