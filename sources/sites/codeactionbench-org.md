# CodeActionBench 项目页

> 来源归档（site）

- **标题：** CodeActionBench: Evaluating Agentic Code-as-Policy for Embodied Manipulation
- **类型：** site（项目主页 + 交互式结果/轨迹展示）
- **URL：** <https://codeactionbench.org/>
- **论文：** <https://arxiv.org/abs/2609.33807>
- **代码：** <https://github.com/lyhkk/CodeActionBench>
- **机构：** 南洋理工大学（NTU）；独立研究者
- **入库日期：** 2026-10-15
- **核查日期：** 2026-10-15
- **一句话说明：** 展示 agent 用视觉观察生成、执行并修订机器人代码的 25-task 操作基准，含演示轨迹、leaderboard 与 agent arena。

## 开源核查（2026-10-15）

| 资源 | 状态 | 说明 |
|------|------|------|
| 代码 | **已开放** | 官方 GitHub 仓库，MIT 许可证；含任务定义、参考 harness、运行工具、示例与文档 |
| 任务配置 | **已公开** | 仓库包含 `benchmark/tasks/` 与配置目录 |
| 评测轨迹数据 | **待发布** | README 标注完整数据集（trajectory、observation images、videos）coming soon |
| 网页演示 | **已开放** | 项目页提供演示、结果表和 head-to-head arena |

**判定：部分开放。** 公开实现和网页演示可检查；完整实验轨迹包尚未发布，不能把这理解成所有论文分析数据均已可下载。

## 页面核查要点

- 页面将任务数、agent 配置数和 episode 数写为 25 × 9 × 3 = 675。
- 项目页结果表按 hidden verifier 的物理成功率排名，另展示 coverage、三次全成功任务数、时间和工具调用数；时间与调用数是描述性资源指标，不参与排序。
- 示范轨迹展示 agent 基于图像估计物体位置、写并执行代码，观察执行结果后调整抓取/放置动作。
- 结果适合比较完整 agent stack；若比较模型本身，应优先注意论文中使用共享 reference harness 的配置。

## 对 wiki 的映射

- [CodeActionBench 论文实体](../../wiki/entities/paper-codeactionbench.md)
- [CodeActionBench 论文来源](../papers/codeactionbench_arxiv_2609_33807.md)
- [CodeActionBench 仓库来源](../repos/codeactionbench.md)
- [操作任务](../../wiki/tasks/manipulation.md)
