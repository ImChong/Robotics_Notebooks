# Embodied Turing Machines: Stateful Code for Robot Recursive Self-Improvement

> 来源归档（ingest）

- **标题：** Embodied Turing Machines: Stateful Code for Robot Recursive Self-Improvement
- **类型：** paper / robotics / code-as-policy / bimanual-manipulation / evaluation
- **arXiv：** <https://arxiv.org/abs/2610.12369>
- **HTML 全文：** <https://arxiv.org/html/2610.12369v1>
- **版本与日期：** v1，2026-10-08
- **作者：** Kairui Hu、Siyuan Hu、Fangzhou Hong、Zhaoxi Chen、Ziwei Liu
- **作者单位：** 南洋理工大学（NTU）；论文作者信息同时标注 Ropedia
- **项目页 / 官方代码：** 截至 2026-10-10，arXiv 记录和论文全文未列独立项目页或公开代码仓库；COAP 描述的是研究方法与程序库设计，不应据此推断实现已开源。
- **一句话说明：** 提出 Code-Only-as-Policy（COAP）：把机器人策略写成能显式测量、维护环境状态并据此执行动作的可读代码；编码智能体可在离线开发阶段迭代代码，而测试运行时不调用 VLM/VLA。作者在 RoboDojo 的 42 个双臂仿真任务上报告 70.24% 总成功率。

## 核心摘录（面向 wiki 编译）

### 1）问题与表示：把策略代码当作可执行状态机

- **摘录要点：** 论文借用具身图灵机视角，把机器人与环境状态视为持续更新的“纸带”，把策略视作可读、可审查的程序；与每步由神经网络直接输出动作不同，COAP 要求代码显式重测状态、维护任务状态，并选择下一步动作。
- **对 wiki 的映射：** [COAP 论文实体](../../wiki/entities/paper-coap-embodied-turing-machines.md) — 代码策略的表示边界与运行结构。

### 2）状态估计与跨任务共享代码

- **摘录要点：** 系统从相机观测、机器人本体状态与已知几何中构造机器人、物体、环境、关系和任务状态；代码在遮挡等情形下保留状态，并能按需再次测量。论文报告新任务中约 83% 的代码可复用，表明复用的是任务程序和操作/感知例程，而不是一个在线生成动作的大模型。
- **对 wiki 的映射：** [RoboDojo](../../wiki/entities/robodojo.md) — COAP 报告采用其 42 项双臂仿真任务；[具身评测基准选型闭环](../../wiki/overview/hub-embodied-eval-benchmark.md) — 仿真任务成功率的解释边界。

### 3）离线自我改进与模型免调用推理

- **摘录要点：** coding agent 的角色属于离线开发环：检查任务轨迹和失败、编辑或扩展共享代码库、通过仿真评测回归；部署后的策略由固定代码读取状态并闭环控制。论文称运行时不需要调用模型。这是“模型辅助开发、代码执行策略”的架构，不代表论文公开了可直接下载运行的仓库。
- **对 wiki 的映射：** [COAP 论文实体](../../wiki/entities/paper-coap-embodied-turing-machines.md) — 离线开发与运行时的分离；[RoboDojo](../../wiki/entities/robodojo.md) — 评测环境与榜单。

### 4）评测与作者报告的结果

- **摘录要点：** 在 RoboDojo 42 个双臂仿真任务、每任务 3 个随机种子与每种子 50 个布局的评测设置下，COAP 报告总成功率 70.24%、progress score 75.45。各维度为 Memory 89.9%、Precision 75.1%、Open 64.1%、Generalization 65.7%、Long-Horizon 56.4%。作者报告其相对当时最佳 Agent Harness 基线 PhysicalRSI（31.38%）提高约 38.86 个百分点。
- **对 wiki 的映射：** [COAP 论文实体](../../wiki/entities/paper-coap-embodied-turing-machines.md) — 指标口径、比较对象及仿真限制；[RoboDojo](../../wiki/entities/robodojo.md) — 统一基准任务背景。

## 资料开放状态核查（2026-10-10）

- **项目页：** arXiv 记录未提供独立项目页。
- **代码仓库：** 论文记录和全文未链接官方可运行代码仓库；截至核查日期按“未提供公开实现入口”记录，不将方法描述等同于开源。
- **评测环境：** RoboDojo 基准本身有独立官网和公开评测代码；这不表示 COAP 策略源码已发布。

## 参考来源

- [arXiv 摘要与版本记录](https://arxiv.org/abs/2610.12369)
- [arXiv HTML 全文 v1](https://arxiv.org/html/2610.12369v1)
- [RoboDojo 官网与评测入口](https://robodojo-benchmark.com/)
- [RoboDojo 论文归档](./robodojo_arxiv_2607_04434.md)

## 对 wiki 的映射

- 主实体：[COAP：Embodied Turing Machines](../../wiki/entities/paper-coap-embodied-turing-machines.md)
- 评测基准：[RoboDojo](../../wiki/entities/robodojo.md)
- 评测选型：[具身评测基准选型闭环](../../wiki/overview/hub-embodied-eval-benchmark.md)
