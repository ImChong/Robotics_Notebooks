# Embodied First Exam — arXiv:2611.12177

- **中文标题：** 具身第一考：机器人环境中的代码智能体能力测评
- **论文：** <https://arxiv.org/abs/2611.12177>
- **HTML 原文：** <https://arxiv.org/html/2611.12177v1>
- **项目：** <https://www.embodied-first-exam.ai/>
- **源码：** <https://github.com/mll-lab-nu/embodied-first-exam>
- **核验日期：** 2026-11-13（复核）；此前已通读记录：2026-11-11
- **核对范围：** arXiv:2611.12177v1 的正文 §1–5、Tables 1–2、Figures 1–18，附录 A–K（任务套件、策略与运行环境、评判、agent 设置、案例、干预、标签体系、真机演示、局限、相关研究），并对照官方仓库 2026-11-11 README；论文统计与后续 README 扩增数据分别核对。
- **通读：** 已完成

## 方法与协议

- Coding agent 在机器人环境里编写控制程序，按允许的观测获得反馈，最终由独立 grader 判断是否完成任务。
- **Standard** 为核心感知与动作设置；**Privileged** 允许更丰富状态与重置；**Realtime** 关注在线实时交互。设置不同，结果不应混为一谈。
- 论文统计 **35 套件、1,156 任务、47 种机器人**，论文核心子集 **456 项任务**；最佳受测 agent 在 Standard 核心任务的成功率 **34%**（论文报告）。

## 版本口径警示

2026-11-11 官方仓库 README 已列出 **36 benchmark / 1,211 unique tasks / 2,398 task directories**，核心配置 **498 tasks**；而 README 明确写论文和 leaderboard 统计为 **35 suites / 1,156 tasks / 2,288 tests**（不含后续纳入的 HumanCLAW）。因此不能用新仓库计数替换旧论文结果的分母。旧 wiki 的 32/449/2128 已过时，需在主节点保留论文与当前代码版本双口径。

## 限制

单次尝试评测可能低估多轮试错能力；一些任务没有完整可直接运行的参考解；模拟任务占主导，不能据此推断真实机器人部署成功率。

## 全文核验补充（2026-11-11）

### 评测协议、分母及成功率

- **论文 35 套件、1,156 独立任务、2,288 个 tests、47 种机器人**，其中九个新套件；核心集 456 任务、901 个测试（Standard 456、Privileged 391、Realtime 54）。任务与 mode 下的测试不能混作独立任务。
- GPT-6 Astra 在核心 Standard 任务上的 **33.6%** 常四舍五入为 **34%**；一模型一 harness 每 test 只有一次尝试，不能当作重复试验平均。
- 在同时有 Standard 和 Privileged 的 391 个核心任务上，成功率在 Standard 约 **28%**、加物体真值 **45%**、加无限 reset **53%**、同时加两者 **68%**，原 Privileged **77%**。这是模型/任务配对实验，不应与 456 Standard 全量 33.6% 混淆。
- 评估采用独立 verifier：Privileged 将提交轨迹复演，Standard/Realtime 由服务端对实跑 episode 计分；agent 无权读取 verifier 或标准/实时环境的私有成功条件。

### 附录与风险

- 对 427 个有阶段标签的任务建立失败分类；更精细的接触姿态任务在 Standard 比 Privileged 难，不能把关联分析当因果结论。
- 392 个任务没有参考解，因此并未证明它们对每个 agent 都可解。各模型仅一次尝试，harness、过滤器、重置权限、时间预算均影响结果。
- Appendix I 的双 agent 真机 Buzz-wire 演示每个 agent 8 trials，**不计入论文仿真成绩**。
- 2026-11-11 官方仓 README 新增 HumanCLAW，报告 36 套件、1,211 任务、核心 498；与论文 snapshot 不同，不要覆盖论文数值。

## 论文正文与附录复核（2026-11-13）

### 评测协议与任务覆盖（§2、Figure 3、Table 2、Appendix A–C）

- 作者采用 Harbor/Terminal-Bench 容器任务格式，固定场景、任务说明、资源限制和外部 verifier；agent 可选择 direct control、code to control、train a policy 或混合，不是把这几种方式规定为不同难度的评测设置。
- **Standard：** 仅可经机器人 API 获得本体/视觉传感信息、发送 action chunks；agent 请求间仿真暂停。**Realtime：** 在 Standard 的观测/重置限制下，仿真在 agent 推理时不暂停，重计算仿真可使用固定速率倍数。**Privileged：** 代理可读完整仿真状态、无限重置，在隔离 verifier 中重放提交的轨迹或执行控制器；所谓“上界”仅是群体近似，不是逐个任务的严格上界。
- 论文版 35 套件 = 新设计 **9 套件/411 任务** + 适配 **26 套件/745 任务**；共 1,156 tasks / 2,288 task-setting tests / 47 robots。核心 **456 tasks / 901 tests**，对应 Standard 456、Privileged 391、Realtime 54。
- 新增九套件按能力分组：接触丰富操作（RoboPaint、RoboARC、DexHandExam），动态任务（RoboDynamicExam、Embodied Olympics - Fun / Match），空间、安全和迁移（iSpatialGym、Embodied Safety Exam、RoboEngineer）。

### 结果与消融（§3.1–3.5、Figures 7–18、Appendix D–H）

- **456 核心 Standard tests：** GPT-6 Astra + Codex **33.6%**（约 34%）。同一批支持两种设置的 **391 tasks**：原始 Standard **30%**、Privileged **77%**；两者都是任务口径配对而不是所有 456 个任务。
- **受控消融（391 tasks）：** 额外运行的 Standard 基线 **28%**，增加物体真值 **45%**，增加无限重置 **53%**，两者同加 **68%**。此处 28% 与上条原始 30% 由不同试验产生，不能混合。
- **54 对 Standard/Realtime：** 最强 agent 完成 **23 / 17**；全部八 agent 均未解的测试占比分别 **50% / 63%**。
- **Harness 敏感性：** GPT-6 Astra 在 Standard 核心集使用 Codex **34%**、Claude Code **22%**。更换执行壳/工具接口可改变分数，不应将榜单视为纯模型排名。
- **阶段失效：** 427 个核心任务有阶段图，作者用另一个 agent 依据回放与日志给失败阶段标注。材料变更阶段失效率 **66%**、机器人身体运动阶段 **45%**；套件/机器人同时变化，不能据此推断阶段类型直接引发失败。
- **控制方式：** 1,489 次抽样尝试中 Standard **61%** 以动作块直接反馈控制，Realtime **97%** 采用程序化控制。足球任务提供 GPU 与 RL 库时出现训练策略的获胜样例，但 CPU/GPU 任务版本有多项差别，不能断言 GPU 是唯一原因。

### 真机证据与不足（Appendix I–J）

- 真机 buzz-wire 演示只有两种 agent、同一机械臂、每个各 **8 次连续尝试**，仅用以说明接口可接真实机械臂，**完全不计入**模拟评测的成功率。
- 392/1,156 任务无参考解；每 test 每 agent 只试一次，任务筛选先用同系列 GPT-6 Luna 过滤部分已解任务，模型家族和标签辅助模型可能引入偏差。多套件来自公开 benchmark，部分任务存在已公开方案的记忆风险。
- 评测取决于具体模型 + harness、任务资源与准入权限；仿真评测不等于真实机器人大规模的鲁棒性或安全能力验证。

### 开放材料与快照

- 论文：[arXiv v1 PDF](https://arxiv.org/pdf/2611.12177) / [HTML](https://arxiv.org/html/2611.12177v1)，项目页：[embodied-first-exam.ai](https://www.embodied-first-exam.ai/)。
- 官方 [MIT 许可代码与任务](https://github.com/mll-lab-nu/embodied-first-exam)、[数据轨迹](https://huggingface.co/datasets/MLL-Lab/embodied-exam-data)；各上游模拟器/资产有额外许可。
- 2026-11-13 官方 README 显示 36 benchmark、1,211 tasks、2,398 task directories、核心 498 tasks。作者明确此仓库计数包含 HumanCLAW；论文及榜单 **35/1,156/2,288** 不包含该套件。新旧快照均需保存。

## 对 wiki 的映射

- [Embodied First Exam 主节点](../../wiki/entities/embodied-first-exam.md)
