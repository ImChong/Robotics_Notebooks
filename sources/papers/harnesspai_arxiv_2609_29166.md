# HarnessPAI: An Evolving Harness for Physical AI（arXiv:2609.29166）

> 论文来源归档；以论文及作者维护的项目页为准。

- **论文：** <https://arxiv.org/abs/2609.29166>
- **HTML：** <https://arxiv.org/html/2609.29166>
- **PDF：** <https://arxiv.org/pdf/2609.29166>
- **项目页：** <https://darwin-agent.github.io/HarnessPAI/>
- **官方 GitHub：** <https://github.com/Darwin-Agent/HarnessPAI>
- **作者团队：** Darwin Agent Team
- **论文日期：** 2026-09-24（arXiv 页面）
- **作者单位：** Xiaomi Inc.; Tsinghua University; Nanjing University; Shanghai Jiao Tong University; East China Normal University; University College London; King's College London; Imperial College London; Institute of Automation, Chinese Academy of Sciences; Shanghai Innovation Institute; University of Oxford; University of Edinburgh; Nanyang Technological University.
- **论文状态：** arXiv 预印本
- **研究代码状态：** 2026-10-08 核验：官方 README 标注代码仍在整理、尚未公开；安装与复现说明尚未提供。当前仓库内容为论文/项目页配套材料，不是可安装的研究框架。
- **入库日期：** 2026-09-26；2026-10-08 更新状态与方法/评测核对
- **项目节点：** [HarnessPAI](../../wiki/entities/paper-harnesspai.md)

## 摘要要点

HarnessPAI 是模型与 embodiment 无关的 Physical AI harness，以代码作为组织动作原语的可执行、可演化接口。它把过程拆成两种时间尺度：

1. **单次 rollout 内：** 执行固定程序，编排感知、几何推理、动作原语和预定义检查。程序级“open-loop”并不表示盲执行；仍可读取观测、进行反馈控制和恢复。
2. **rollout 之间：** coding agent 根据诊断和轨迹定位失败、修订隔离的候选程序并验证；验证通过后写入可复用代码记忆，并将修复经验整理为结构化技能记忆。

任务记忆用于复用既有程序；原语库提供感知、控制及学习型动作能力。演示视频工作流抽取、动作模型能力评估、感知模块初始化是可选准备步骤。

## 论文报告的主要结果

- LIBERO-PRO 三套件 × Swap/Task：相对 π₀.₅-LIBERO 从 34.9% 到 96.5%，+61.6 个百分点。
- RoboCasa Target50 Atomic-Seen（18 tasks）：相对 WorldDreamer 从 65.0% 到 92.2%，+27.2 个百分点。
- robosuite 七项操作任务：报告相对 ASPIRE +15.3 个百分点；该参照来自先前工作，不是同后端受控对照。
- LIBERO 三套件：相对 π₀.₅-LIBERO +1.2 个百分点。
- 额外迁移和后训练实验应与冻结动作后端的主结果区分；论文报告的 π₀.₅ 微调 +38.8 个百分点属于独立下游实验。
- 七类设置使用不同指标；如 VacuSim 覆盖率和 MicroDuck 横向误差不可合并成统一成功率。

## 评测边界

论文说明 LIBERO 与 LIBERO-PRO 每项任务评估 50 个 seeds，其中 15 个用于程序演化，另 35 个用于评估；RoboCasa 比较使用同一 20-seed 设置。项目页将 robosuite 的 ASPIRE 标为先前工作数据。复现或横向比较前应核验具体套件、任务划分、成功条件与程序演化预算。

## 项目定位

这是论文/项目材料归档，不是代码复现说明。官方 GitHub README 在 2026-10-08 的状态声明是：项目首页与演示、论文 PDF 可访问；研究代码仍在整理，尚未公开；安装与复现指引尚未提供。请查看[仓库状态归档](../repos/harnesspai.md)与[项目页归档](../sites/harnesspai.md)。
