# Towards Agentic Robots: Model, Data, Environment, and Harness（迈向智能体机器人：模型、数据、环境与编排）

> 原始论文资料归档；主身份 arXiv:2610.20442，固定 **v1**，不与持续更新的 Awesome 列表混用。

- **标题：** Towards Agentic Robots: Model, Data, Environment, and Harness
- **中文标题：** 迈向智能体机器人：模型、数据、环境与编排
- **作者：** Chonghan Qin、Dongming Wu、Kaituo Feng、Jiaxu Wang、Xinlei Yu、Peiwen Sun、Yijia Fan、Xilin He、Bingao Chen、Weitong Lian、Ryan Ji、Wenqi Ren、Xiaobin Hu、Dahua Lin、Yu-Gang Jiang、Xiangyu Yue
- **机构：** MMLab / 香港中文大学、复旦大学、NovaXBot、清华大学、浙江大学、中山大学（论文首页列出合作机构，未按作者逐一对应）
- **类型：** paper / survey / agentic-robots / robotics / vla / world-model / harness
- **arXiv 信息：** [摘要 v1](https://arxiv.org/abs/2610.20442v1) · [HTML 正文 v1](https://arxiv.org/html/2610.20442v1) · [PDF v1](https://arxiv.org/pdf/2610.20442v1) · [TeX 源包 v1](https://arxiv.org/e-print/2610.20442v1)
- **日期：** arXiv 首投 2026-10-14（v1）；本次核查 2026-11-16
- **配套文献仓：** [chonghanqin/Awesome-Agentic-Robots](https://github.com/chonghanqin/Awesome-Agentic-Robots)，本库归档：[sources/repos/awesome-agentic-robots-survey-index.md](../repos/awesome-agentic-robots-survey-index.md)
- **身份与开放范围：** 论文是综述，不提供本文专属训练、推理或部署算法的可运行实现；配套仓库公开了文献索引与生成/校验脚本，**不是机器人控制代码**。
- **核对范围：** 2026-11-16 对照 arXiv:2610.20442v1 HTML 正文逐节阅读 §1–§9：§1 研究动机；§2 传统策略 / VLA / WM-WAM / agentic 四条演化脉络；§3 定义、Figure 3 与 4 域 17 类；§4 Training、Perception、Planning、Robot Control、Adaptation；§5 Collection、Annotation、Refinement；§6 Reconstruction、Task Generation、Benchmarking；§7 Orchestration、Memory、Monitoring and Recovery、Skills、Infrastructure、Layered Systems；§8 四项展望；§9 结论。逐节对照 HTML 的表 1–18 及图注中的类别、接口、实验对照口径；先前另已核对 v1 PDF Figure 1 / Figure 2 / Figure 7 及第 10 页 Table 1 的排版；配套文献仓 README 与目录单独核对。References 列表只抽样检查高关联条目，不逐篇复核所有被引研究。
- **通读：** 已完成
- **核对限制：** 「通读：已完成」仅指 arXiv v1 HTML 正文 §1–§9、主要表格与图注，**不是** TeX 源代码审计、79 页 PDF 逐字比对或所有参考文献逐篇核查。TeX e-print URL 已保存，但未成功解包核验公式源码；官方 Awesome 仓库持续更新，不可逆向归入论文 v1。本论文是综述，没有可复现的统一模型训练或真机实验。
- **入库日期：** 2026-10-15；本次深化 2026-11-16
- **对 wiki 的映射：** [唯一论文实体：paper-towards-agentic-robots-survey](../../wiki/entities/paper-towards-agentic-robots-survey.md)

## 摘录 1：定义、范围和发展脉络（§2–3）

论文将 Agentic Robots 定义为：智能体根据目标、环境观测与执行反馈，自主规划并协调机器人任务；选择动作、技能或工具，持续监控进展，并在失败时获取新信息、修订计划或恢复。范围包含仿真与物理机器人，并不意味着「绝对无需人介入」或自动高于传统自主机器人。

文中 Figure 1 的四个重叠范式：传统机器人策略 → VLA → World Model / WAM → Agentic Robots。作者强调不同范式可作为同一系统中的互补能力，而非简单取代关系。

## 摘录 2：Table 1 四域十七个**非互斥**功能类别（§3.2，PDF 第 10 页）

表中定义按「智能体决定改变系统的哪个对象」组织；一个工作可以出现于多个类别。

| 域 | 类别（原文） | 按 Table 1 的中文释义 |
|---|---|---|
| Model | Training | 设计或修订奖励、优化目标、训练配方与学习实验 |
| Model | Perception | 根据观测恢复任务相关概念、关系、几何或信念，并判断是否需要补充证据 |
| Model | Planning | 随任务进展选择、修订子目标、动作序列、依赖或任务方案 |
| Model | Robot Control | 将操作转为机器人可执行的动作、姿态、程序或运动约束 |
| Model | Adaptation | 利用经验修改模型参数、任务条件化信息或决策规则 |
| Data | Collection | 获取或生成用于学习或后续使用的新机器人经验 |
| Data | Annotation | 解释已有经验并增加文本、空间、交互目标或结果等结构化监督 |
| Data | Refinement | 筛选有用片段、修订不可靠监督，提高既有经验质量 |
| Environment | Reconstruction | 构建或修改可执行场景、资产、几何、任务相关物性 |
| Environment | Task Generation | 定义或生成目标、约束、课程或任务分布 |
| Environment | Benchmarking | 构造或调整测试、扰动、指标、评估器与评测流程 |
| Harness | Orchestration | 选择模型/策略/技能/工具并管理调用与交接 |
| Harness | Memory | 保存、更新、检索用于后续决策的交互历史或知识 |
| Harness | Monitoring and Recovery | 评估进度、完成、失败与约束，并组织纠错响应 |
| Harness | Skills | 创建、验证、保存、修订或淘汰可复用程序知识及可执行行为 |
| Harness | Infrastructure | 经接口、服务和运行时检查暴露、提供、门控工具与策略 |
| Harness | Layered Systems | 在规划器、学习策略、解析控制间分配信息、决策权限与运行频率 |

## 摘录 3：四域如何协作（§4–7）

- **Model（§4，5 类）：** 不只是动作预测；智能体可修订训练环、主动解释观测、生成/检查/更新计划、选择动作接口及根据经验适应。
- **Data（§5，3 类）：** 采集物理/仿真/合成数据 → 加文本/空间/交互/结果标注 → 通过反馈、筛选和纠错精炼数据；别把数据越多直接当作训练必然越好。
- **Environment（§6，3 类）：** 重建可执行世界、生成可解任务、设计可靠的 reset/指标/扰动等评测；视觉相似不等于可执行物理一致性。
- **Harness（§7，6 类）：** 模型、工具和策略的选择与交接；维护记忆和技能；监测失败并恢复；接口/服务/运行门控；按不同频率和权限切分规划、策略及解析控制。Figure 7 重点是目标/证据 → 调度 → 执行器 → 机器人 → 反馈的闭环，而非新的端到端网络结构。
- **论文实例：** HarnessVLA 同时涉及编排、记忆、监测与恢复；Zero2Skill 跨 Data 与 Model；ENPIRE 通过实验组织连接训练与能力改进（§3.2）。

## 摘录 4：区别与评测边界（§3.3、§8）

- **相对 Embodied AI：** 关注如何组织技能、执行任务并依反馈修订，而不只是智能体在环境中学习/行动。
- **相对 Autonomous Robotics：** 规划/监控等能力并非 agentic 独创；论文着重基础模型智能体参与目标解释、技能协调、反馈决策。
- **相对 Robot Foundation Models：** 关注已学习能力在持续任务中的调用与协调；架构可以是单模型或多组件。
- **四项前瞻：** Fair Benchmarking；Real-robot Agentic Robotics；Efficient Robot Use；Cross-Embodiment Generalization。
- **可比性：** 只给成功率无法隔离智能体作用；还需报告底层执行器、工具、额外观测、重试预算、执行时间、计算资源及人类协助。
- **证据级别：** 这是跨研究文献综述，**不是单一可复现实验**；表格中各工作的指标来自不同设定，不能据此宣称统一 SOTA 或证明四域组合必定优于单策略。

## 全文复核补充：Harness 的接口与时序边界（§7.1–7.6）

- **Orchestration（§7.1）**：调用规划器、VLA、可执行技能和工具需要统一输入/输出协议、状态连接与正确的交接时机；上一步完成信号与环境状态确认不可省。
- **Memory（§7.2）**：写入、更新、检索是不同职责；历史记忆可保存阶段进度，但过时的目标位姿应重新感知和校验。
- **Monitoring and Recovery（§7.3）**：将执行前/中/后的监测与重试、改计划、替换执行器、场景复位分开；失败后未必应回到初始状态，重试也必须有预算。
- **Skills（§7.4）**：从代码策略、技能库到迭代改进，只有执行与验证后的行为才能作为可信的复用单元；「生成了代码」不等于「获得了新能力」。
- **Infrastructure（§7.5）**：工具服务、VLA 推理、版本门控与实时安全检查都属于 harness 的实际约束，不能只用 agent 的文本决策替代。
- **Layered Systems（§7.6）**：System 2 做任务/子目标推理，System 1 为学习式 VLA 等执行器，System 0 为 IK / MPC / WBC 等解析控制；不同更新频率的例子包括 GR00T N1 约 10 Hz 高层模块与 120 Hz 动作模块（论文报告的配置），不是统一的硬件控制频率要求。
- **评测与展望（§8）**：如要归因 agent 应固定执行器、额外观察和重试预算，并同时报告端到端时间、推理成本、执行失败及恢复；跨本体泛化应区分重用高层任务策略与重新适配具体驱动/控制接口。

以上是综述**作者对既有系统的归类与讨论**，不是本文新提出并实测的集成实现；详见 [arXiv:2610.20442v1 正文](https://arxiv.org/html/2610.20442v1)。

## 来源版本规则

- **固定快照：** 仅上述 arXiv:2610.20442v1 的摘要、HTML、PDF、TeX URL 支持「本文说了什么」。
- **动态索引：** 作者仓库 main 的文献条目持续变化；用于检索后续工作，不把新增条目或 README 更新倒写成 2026-10-14 的论文论点。
- **源码运行时序图：** 不适用（综述没有可运行的机器人方法源码）；配套仓库的索引脚本属于文献维护工具。

## 对 wiki 的映射

- [论文实体页](../../wiki/entities/paper-towards-agentic-robots-survey.md)：按 Table 1 展开四域、纠正「Model=训练/感知/规划」的遗漏，补上具体编排和评测协议边界。
- [VLA 纵深路线](../../roadmap/depth-vla.md)：Stage 4 部署与系统整合，作为策略/agent/harness 分工的综述入口。
