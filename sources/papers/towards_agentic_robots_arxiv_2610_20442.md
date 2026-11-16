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
- **核对范围：** arXiv v1 HTML 第 1–9 节的定义、分类、四功能域、展望与结论；PDF 首页 Figure 1、正文 Figure 2 / Figure 7 图注及第 10 页 Table 1 排版和 17 项定义；作者配套仓库 README / 目录。
- **通读：** 未完成
- **核对限制：** TeX e-print 链接已保存但当前环境无法解包读取，故未独立复核 TeX 公式源码与原始图注；文献综述的全部逐项引文及 79 页 PDF 中的每条引用尚未一一复核。本记录只对已实际核对的论文主框架和 Table 1 给出结论。
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

## 来源版本规则

- **固定快照：** 仅上述 arXiv:2610.20442v1 的摘要、HTML、PDF、TeX URL 支持「本文说了什么」。
- **动态索引：** 作者仓库 main 的文献条目持续变化；用于检索后续工作，不把新增条目或 README 更新倒写成 2026-10-14 的论文论点。
- **源码运行时序图：** 不适用（综述没有可运行的机器人方法源码）；配套仓库的索引脚本属于文献维护工具。

## 对 wiki 的映射

- [论文实体页](../../wiki/entities/paper-towards-agentic-robots-survey.md)：按 Table 1 展开四域、纠正「Model=训练/感知/规划」的遗漏，补上具体编排和评测协议边界。
- [VLA 纵深路线](../../roadmap/depth-vla.md)：Stage 4 部署与系统整合，作为策略/agent/harness 分工的综述入口。
