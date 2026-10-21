# APTree: Extended Behavior Trees for Adaptive and Scalable Automated Robotic Task Planning in Construction（arXiv:2610.18523）

> 来源归档（ingest）

- **标题：** APTree: Extended Behavior Trees for Adaptive and Scalable Automated Robotic Task Planning in Construction
- **arXiv：** <https://arxiv.org/abs/2610.18523>（HTML：<https://arxiv.org/html/2610.18523v1>），cs.RO / cs.AI / cs.SE，2026-10-13 提交；已被 The Journal of Systems and Software 接收；46 页，17 图，6 表
- **代码：** <https://github.com/shshermin/APTree-automated-planning-tree>（[sources/repos/aptree.md](../repos/aptree.md)）
- **项目页：** 论文与仓库未见独立项目页（2026-10-17 核查）
- **作者：** Shermin Sherkat、Thomas Wortmann、Andreas Wortmann；斯图加特大学 ICD/CA（计算设计与建造研究所）；Sherkat 与 A. Wortmann 另属 ISW（机床与制造单元控制工程研究所）
- **入库日期：** 2026-10-17
- **一句话说明：** 模型驱动框架：DSL 描述任务，运行时协调多个符号规划器生成模块化行为树，并支持局部重规划。

## 开源状态（步骤 2.5）

- **无项目页**；论文 HTML 称输入模型在 Git 仓库（URL 在已读部分未出现，Appendix C 未读到）。
- **仓库核查（2026-10-17）：** 公开仓库，273 次提交，1 star；含 `APTreeDSL/`（MontiCore + Gradle）、`APTreeExecutionEngine/`（C# .NET）、`APTreeEditor/`、`InputGenerator/`、`docker/`、`Dockerfile`、`docker-compose.yml`、`DOCKER_README.md`。页面未显示许可证。
- **结论：** **已开源**（DSL、执行引擎、Docker 运行说明可见）；本次未在本地构建运行。

## 摘录 1：方法

- 扩展行为树元模型，含四个包：领域类型、基于图的任务计划、规划服务、动态流程节点；共享 blackboard，节点状态新增 Uninitialized 与 ReadyToTick。
- 文本 DSL 用 MontiCore 实现，负责校验并为执行引擎生成代码；执行引擎加载模型、调用符号规划器。
- 规划器：高层 ENHSP，中层 Metric-FF 与 LAMA-First（PDDL 域文件 2 个：高层、中层）。
- 局部重规划只替换受影响的计划模块；定义结构变化、重规划频率、节点保留率等指标。
- 摘要称有机器校验的分析：在规划器可靠（soundness）与计划互不干扰等假设下，组合执行与局部重规划安全。校验工具名与 Appendix B 内容未读到（**未核实**）。

## 摘录 2：livMatS Biomimetic Shell 案例（Section 6–7）

- 12 个 cassette，分 3 个生产批次，每批 4 个；每个 cassette 6 根梁、2 块板；12 个初始问题文件。
- 最终执行模型 1,149 个动作节点，深度 6；高层动作实例 372；重规划下共生成 6,507 个动作实例、1,098 次重规划事件，PDDL 问题文件 1,470（无重规划 372）。
- 无重规划：执行 5 分 06 秒，规划延迟 2 分 20 秒；有重规划：执行 13 分 29 秒，规划延迟 8 分 53 秒；三规划器配置规划延迟 43 分 08 秒（总 48 分，89.9%），单次调用均值约 1,746 ms（ENHSP + Metric-FF 约 360 ms）。
- 注入错误检出（Table 3）：APTree TPR 76%，PlanSys2 25%；类型错误 APTree 100%、PlanSys2 0%；结构错误两者 100%（阶段不同）；逻辑错误一半自动解决、一半中止执行。
- 局部重规划：节点变化均值 0.80%（0–6.50%），PlanSys2 为 100% 重生成；宽度变化 0.38%；重规划频率 81.4 次/分；节点保留率 17.7%；树深保持 6；372 个高层任务均达成目标。
- 局限（Section 8.2）与复现细节（Appendix C）未读到。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-aptree.md`。
