# AgenticGenPlan：代码代理合成可泛化的任务与运动规划程序（arXiv:2609.30233）

> 一手论文阅读归档；与项目网站、官方源码分开保存。不是对任意仿真器之间零样本迁移的证据。

- **原题：** Coding Agents for Generalized Task and Motion Planning Problems
- **作者：** Matteo Merler、Bowen Li、Josh Roy、Yichao Liang、Qianwei Wang、Yixuan Huang、Tom Silver
- **机构：** Fondazione Bruno Kessler、Carnegie Mellon University、Princeton University、University of Cambridge
- **版本：** arXiv:2609.30233v1，2026-09-24
- **论文：** https://arxiv.org/abs/2609.30233
- **HTML 全文：** https://arxiv.org/html/2609.30233v1
- **PDF：** https://arxiv.org/pdf/2609.30233
- **项目页：** https://agenticgentamp.github.io/
- **官方代码：** https://github.com/tomsilver/robocode
- **作者公开运行数据：** https://huggingface.co/datasets/prpl-group/agentic-gentamp-runs
- **核对日期：** 2026-11-10
- **核对范围：** 通读 arXiv 官方 HTML v1 的 Abstract、§I Introduction、§II-A–C（MDP 与泛化定义）、§III-A–B（代理配置、隔离条件与基线）、§IV-A–B（实验协议、Tables I–III、Figures 1–4 的文字说明、Q1–Q5）、§V-A–D（关联工作）、§VI（局限/结论）及参考文献；并交叉核对作者项目页与开源仓库 README。正文没有独立附录。全文阅读依据官方 HTML，非只读摘要。
- **通读：** 已完成

## 研究问题与定义（§I–II）

本文研究 Generalized Task and Motion Planning（泛化 TAMP）。一个环境具有共同的状态/动作空间、状态转移与奖励函数，不同**问题实例**在物体数量、物体位置、几何配置上变化。代理不是把一条固定动作序列复用，而是在固定预算内编写一个以状态为输入、输出动作的**程序策略**，用于该环境内未见过的初始实例。

状态为**完全可观测的对象中心特征**，无需视觉感知与自然语言理解；训练阶段可以调用仿真 reset、step、render，测试阶段冻结程序、不再调用 LLM。论文**没有证明**将某环境的程序直接用于另一个动力学/API/任务定义不同的环境。

论文的策略定义为 `a_t = π(h_t)`，其中 `h_t = (s_0,a_0,…,s_t)`，程序可以维护内部状态；评价是给定步数与时间预算内到达目标的概率。

## 方法（§III）

1. **主实验严格黑盒：** 环境描述、状态/动作定义、目标，以及独立进程提供的 reset/step/render 客户端；看不到模拟器源码、预制 TAMP 谓词、算子、采样器或技能。代理在隔离 Docker 内使用 Python + NumPy + SciPy，不能访问网络/宿主机。
2. **自主迭代：** 编码代理选择探索动作、编写并执行测试脚本、观测物理/几何反馈、发现角落样例，修正 `GeneratedApproach` 程序。每次测试前记录 Git 提交以便审计演化。
3. **固定预算与交付：** 每环境每次合成模型使用上限 20 美元；产物实现 reset/get_action，封存后在新种子上复用，评估时无 LLM 请求。
4. **+ source 消融：** 允许 Claude Code / Codex 看到环境源码、导入辅助函数（如 IK），还能构造仿真状态，用于衡量完整环境知识的增益。
5. **对照：** 16 环境提供人工工程化 TAMP planner（KinDER/SeSamE 或 PDDLStream）；另对比给源码但不能主动运行工具/选择测试的 LLMGenPlan，以及首轮生成的 one-shot。

## 评测协议与结果（§IV；Tables I–III）

- **环境：** 25 个 KinDER（Kinematic2D、Dynamic2D、Kinematic3D、Dynamic3D）+ 3 个 PDDLStream（Packing、Blocked、Rovers）；19 个 KinDER 环境具有变化物体数量的变体，部分计数超出原基准。
- **规模：** 7 种程序合成条件 × 28 环境 × 每条件 5 次独立合成 = 980 个程序；每程序 100 个保留测试实例，总计 98,000 次评估。相同环境中各方法采用相同保留测试实例；每个实例超时为 60 秒。
- **16 个有人工 planner 的环境上的平均成功率：** planner 47%；Claude Code / Opus 5 82%；Codex / GPT-5.6 Sol 56%；Codex / GPT-6 Astra 95%。Astra、Opus、Sol 分别在 15、12、9 个环境上优于 planner；均值**不等于**每个任务都更好。
- **28 环境全部平均：** Opus 黑盒 74%，看到源码后 84%；Astra 黑盒 86%，看到源码后 95%；同样看到源码但没有自主测试的 LLMGenPlan 为 28%（比较需注意工具/测试/反馈机制差异）。
- **困难环境：** Astra 在 Dynamic3D 家族平均仅 65%；SweepSimple、ScoopPour 和接触复杂环境依然失败较多；主设置不等于鲁棒跨任务通解。
- **计算开销：** 具有不同物体数量且有 planner 的 14 环境中，每实例策略计算耗时 Opus 约 2.1 s、Astra 约 0.5 s、planner 约 29 s。另一组**15 个环境中存在全成功 run**的严格筛选子集，平均每动作决策耗时为 Opus 黑盒 11.741 ms、Astra 黑盒 1.299 ms，+source 分别 42.480/50.392 ms；两种统计**不可混作同一测量口径**。

## 论文中的可核对行为（§IV-B）

- StickButton：自动反复 reset 不同种子，采集贴墙杆/高按钮边界情形；先侧抓移杆，再从下方重抓。
- Shelf：借助抓住的立方块作间接末端标记，实测校准机械臂模型，再用于 IK。
- Blocked：有些程序在近处目标失败时切换到远端可抓物体；一次 Sol 提交使成功率从 15% 升到 56%，后续达到 83%。
- SweepIntoDrawer：某些程序改用夹爪逐个取块，而不是按照任务预期依赖扫帚；这是特定仿真任务中的可行行为，不代表现实中一定安全/适合。

## 局限、风险与误读（§VI）

- **泛化边界：** 单个共享 MDP/接口的不同实例，并非直接跨 KinDER ↔ PDDLStream、跨机器人模型、跨动力学引擎或任意真实场景迁移。
- 完全观测的对象状态规避了视觉识别、传感噪声、模型误差和部署时延；没有真机 Sim2Real 测试或安全认证。
- 未公开模型训练数据，不排除与部分公开 benchmark 源码/模式训练数据重合的可能；作者以迭代日志与意外策略提供额外证据，而非形式证明。
- 有源码可以提高成功率，也可能让生成程序依赖昂贵辅助求解；1.299 ms/动作不是独立真机控制环路延迟保证。
- 评测保留种子由作者隐藏以减轻泄漏；公开代码和 runs 数据不等于自动复现论文相同保留测试样本。

## 对 Robotics_Notebooks 的映射

- [AgenticGenPlan 项目实体页](../../wiki/entities/paper-agenticgenplan-generalized-tamp-coding-agents.md)
- [作者项目页归档](../sites/agenticgentamp.md)
- [RoboCode 源码归档](../repos/tomsilver-robocode-agenticgenplan.md)

## 一手资料

1. [作者 arXiv HTML 正文 v1](https://arxiv.org/html/2609.30233v1)
2. [作者项目网站](https://agenticgentamp.github.io/)
3. [官方代码 RoboCode](https://github.com/tomsilver/robocode)
4. [官方运行数据](https://huggingface.co/datasets/prpl-group/agentic-gentamp-runs)
