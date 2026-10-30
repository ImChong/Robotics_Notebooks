---
type: paper
tags: [robot-learning, fleet-learning, manipulation]
status: draft
updated: 2026-10-30
arxiv: "2610.25736"
project_id: collective-experience-learning
summary: "CEL 多机器人经验学习来源待核查：论文全文未获取，待通读后完成 ingest。"
---

# Learning from Fleet Experience: Collective Experience Learning Across 96 Robots — 来源待核查

- **论文：** [arXiv:2610.25736](https://arxiv.org/abs/2610.25736)，v2（2026-10-21）
- **项目主页：** https://cel-robot-learning.github.io/
- **官方代码入口：** https://github.com/RLinf/RLinf — 仓库 README 称 CEL 代码即将发布；截至 2026-10-30 未核实 CEL 可运行实现。
- **机构：** 清华大学、智元机器人、Infinigence AI（项目主页列示）。
- **阅读状态：** 项目主页已阅读；论文全文未获取。arXiv HTML 返回不可访问，PDF 因文件过大未能读取。以下仅为主页可核查线索，不能代替通读论文。

## 项目主页线索

- CEL 在专家监督初始化后汇集车队执行经验更新共享策略；人工仍提供每次任务的成功/失败标签并重置环境。
- FERT 使用失败前的观测检索成功轨迹片段，提高相应成功动作的训练采样权重；FATS 通过异步采集与局部故障恢复支撑训练。
- 主页报告手机包装在 96 台训练机器人、两台未参与采集的同型号机器人的 120 次评估中，成功率从 79.2% 提高到 98.3%。实验设置和统计细节仍需核对论文全文。

## 未完成事项

1. 获取并通读 v2 论文全文和相关附录，核查方法、实验、局限及版本。
2. 复查 CEL 源码是否发布；若存在可运行实现，阅读入口及相关源码并绘制源码时序图。
3. 按仓库规范创建或更新项目实体、基于全文绘制 Mermaid 流程图、添加项目页与代码归档及交叉链接。
4. 运行 `make ci-preflight` 与适用的 Mermaid 检查后再将 PR 标记为可评审。
