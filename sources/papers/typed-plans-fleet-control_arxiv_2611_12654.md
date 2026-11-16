# Correctness Boundaries of Typed Plans for Natural-Language Robot Fleet Control（arXiv:2611.12654）

> 来源归档（paper）

- **标题：** Correctness Boundaries of Typed Plans for Natural-Language Robot Fleet Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.12654>
- **PDF：** <https://arxiv.org/pdf/2611.12654v1>（v1，2026-11-09，cs.RO）
- **作者 / 机构：** Artem Tuzov 等 6 人，Innopolis University（俄罗斯）
- **代码：** https://github.com/Innopolis-Robotics-Society/llm_swarm
- **项目页：** <https://innopolis-robotics-society.github.io/llm_swarm/>（跳转到 Sphinx 文档站 iros_llm_swarm_docs，未逐页核对）
- **入库日期：** 2026-11-11
- **一句话说明：** 在 20 机器人 ROS 2 / Stage 仿真栈上追踪「语言指令 → 类型化计划 → 确定性修正 → 车队协调 → 完成检查」，说明各层检查与成功报告各自覆盖原始指令的哪些要求。
- **核对范围：** 用 pdftotext 通读 v1 PDF 全文 1–18 页（摘要、引言、相关工作、系统概述、三层实现、实验设计 E1/E2、结果 8.1–8.5、讨论、局限、结论、声明与参考文献）；图 1–3 为示意图，仅读图注与正文描述，未看图像像素；论文称数据在仓库 paper/results 与 135 MB 公共归档，均未下载，也未运行任何代码；仓库 README 与项目页只静态读了首页 / 文档站首页。
- **通读：** 已完成

## 开源状态

- 论文称 ROS 2 栈、基准 harness 与分析脚本在仓库中，原始 bag 在公共结果归档（llm_swarm_results_2026-11-04.tar.zst，135 MB，附 SHA-256）；两条被替换的 B/M2 运行的 bag 缺失（论文第 7.2、Data Availability 节）。
- 仓库 README 称 ROS bags 通过 Google Drive 链接提供；仓库未逐文件核对（2026-11-11）。
- 论文自述：历史仿真运行的未提交改动未被版本记录捕获（Code Availability、第 10 节）。

## 核心摘录

1. **四类正确性问题（第 3 节）：** 结构有效、任务含义、几何可行、评估器完成，论文称类型化表示本身不蕴含它们之间的任何推出关系。
2. **计划文法（第 4 节）：** 叶子 mapf / formation / disband / idle，容器 sequence / parallel 可嵌套；操作员聊天路径经八个白名单工具可查位姿与任务状态。
3. **E1 离线基准（第 7.1、8.4 节，表 3）：** 44 例 12 类，每条件每例 5 次、温度 0.1，每格 220 条记录；本地 5 配置无 schema 通过率 70.0–87.3%。
4. **E2 仿真（第 7.2、8.5 节，表 2、5）：** 选定 18 次 M2/M3 运行全部满足位姿评估器；其中 B/M2 含 2 次因软件改动替换 4/5 停滞运行的成功重跑，且 18 个 worktree 均为 dirty。
5. **案例（第 8.1–8.3 节）：** 额外取货分配、共享卸货点目标微调、列队被生成成横排后路径被拒、MAPF 报告 All agents arrived 却漏掉 robot 0。
6. **评估器（第 5 节）：** 5 Hz 位姿采样，访问任务 1.5 m 内即完成；搬运任务记录先在取货区、后在卸货区检测到同一机器人，物理装卸不在度量内。

## 对 wiki 的映射

- [paper-typed-plans-fleet-control](../../wiki/entities/paper-typed-plans-fleet-control.md)
