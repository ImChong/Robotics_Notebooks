# AgenticGenPlan：代码代理合成可泛化 TAMP 程序（arXiv:2609.30233）

> 来源归档（paper），依据作者论文官方 HTML v1 全文及表 I–III；不将跨实例泛化写成跨任务零样本迁移。

- **原文题目：** Coding Agents for Generalized Task and Motion Planning Problems
- **作者：** Matteo Merler、Bowen Li、Josh Roy、Yichao Liang、Qianwei Wang、Yixuan Huang、Tom Silver
- **机构：** Fondazione Bruno Kessler、Carnegie Mellon University、Princeton University、University of Cambridge
- **类型：** paper / generalized TAMP / coding agent / program synthesis
- **版本：** arXiv:2609.30233v1；2026-09-24
- **论文：** <https://arxiv.org/abs/2609.30233>
- **全文：** <https://arxiv.org/html/2609.30233v1>
- **PDF：** <https://arxiv.org/pdf/2609.30233>
- **项目：** <https://agenticgentamp.github.io/>
- **代码：** <https://github.com/tomsilver/robocode>
- **核对日期：** 2026-11-10
- **核对范围：** 通读官方 HTML v1 的摘要、§I 引言、§II 问题设定（II-A–C）、§III 编码代理与基线（III-A–B）、§IV 实验（IV-A–B，含完整表 I、II、III 与图 1–4 文字说明）、§V 相关工作（V-A–D）、§VI 讨论/局限及参考文献；图形以作者图注和页面展示核对。未逐篇通读被引论文、未在本环境复跑所有 98,000 episodes。
- **通读：** 已完成

## 实验问题与定义（§II）

- 目标不是让一个程序适应所有 28 个不同任务，而是在**同一环境任务族**共享状态/动作空间、转移机制与奖励的前提下，泛化至对象数量、几何及布局不同的新初始状态。
- 环境为完全观测的物体中心状态（含位置、几何、关节与速度），主设置不解决视觉感知、语言落地或真实机器人部署。
- 合成期间允许选择测试种子、反复 reset/step/render、写代码/调试；测试期冻结 `reset/get_action` 程序，不再调用 LLM。

## 方法（§III）

1. 黑盒主设置：Docker 隔离，提供任务描述、observation/action space、目标和仿真 API；不提供环境源码、人工 TAMP 谓词/算子/采样器/技能。仅允许 Python 标准库、NumPy、SciPy。
2. 代理自己做针对性仿真探针、边界案例测试、模型校准与策略修订，并在提交测试前用 git 记录程序变更；每轮合成模型预算为 $20。
3. 附加 +source 条件允许读取仿真源码及复用 IK 等内部例程；不是黑盒结果的一部分。
4. 基线：手工工程 TAMP planner（16/28 环境有提供）、LLMGenPlan（提供源码但不能自定义测试，仅固定反馈）、one-shot（首个程序）。

## 实验协议和数字（§IV）

- 25 个 KinDER 环境 + 3 个 PDDLStream 领域；28 个环境、每方法/环境 5 次独立程序合成、每程序 100 个测试实例。合计 **980 个程序 × 100 = 98,000 个 episode**（包含七种合成方法条件）；测试每实例限时 60 秒。
- **有 planner 的 16 环境**均值：Astra 95%、Opus 82%、Sol 56%、人工 planner 47%；三个配置不是在所有环境都胜过 planner，按环境计 Astra 15/16、Opus 12/16、Sol 9/16。
- **全部 28 环境**：给予环境源码时 Opus 成功率 74% → 84%、Astra 86% → 95%；同获源码的 LLMGenPlan 28%，不能将增益单独归因于有源码。
- Astra 主设置按类别：Kinematic2D 99%、Dynamic2D 97%、Kinematic3D 93%、Dynamic3D 65%、PDDLStream 接近 100%。后者不能掩盖 SweepSimple / ScoopPour 等动态 3D 失败。
- 论文表 III **仅选 15 个在四个 agent 条件均至少有完美 run 的环境**比较动作计算时间：主设置 Astra 1.299 ms/action、Opus 11.741 ms/action；允许源码时分别 50.392 和 42.480 ms/action。**不是**端到端机器人闭环延迟或所有 28 环境的无条件平均。
- 论文另报告在 14 个存在 planner 且支持多种物体数的环境中，平均单实例程序计算 Opus 2.1 s、Astra 0.5 s、planner 29 s；均不含合成时的模型开销。

## 机制观察（§IV-B）

- StickButton：代理主动采样墙边棍、较高按钮等边界条件测试抓取序列。
- Shelf：通过夹着立方体运动并观测其位置校准 Kinova Gen3 几何，6 个安装/抓取偏移参数在校准观测上的 RMSE 从 38.9 mm 降至 1.8 mm；不是实机标定精度认证。
- 生成策略不总是逐实例搜索：也可能硬编码动作、解析几何、局部搜索、失败回退（Blocked 的备用方块）；模型和程序之间存在显著运行差异。
- 更强代理的优势包括持续测试和更稳定的解法生成（如 Shelf 中 Opus 不同运行成功率跨 0–100%）。

## 解释边界（§VI）

- 测试来自**与合成相同的初始状态分布**，但种子不重叠。**不证明在任意其它 simulator/任务/动力学中直接可用**；更换仿真引擎、观测接口、动作语义或接触参数均需重新验证/适配。
- 无感知不确定性评估；模型训练集不可见，不能完全排除先前见过 benchmark 代码的可能；未实机验证。
- 同一任务族的场景/物体数量变动可作为已验证范畴；跨任务通用性不是论文结论。

## Wiki 映射

- [AgenticGenPlan 项目实体](../../wiki/entities/paper-agenticgenplan-generalized-tamp.md)
- [作者项目页归档](../sites/agenticgentamp.md)
- [官方源码归档](../repos/tomsilver-robocode.md)
