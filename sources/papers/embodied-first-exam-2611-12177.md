# Embodied First Exam — arXiv:2611.12177

- **中文标题：** 具身第一考：机器人环境中的代码智能体能力测评
- **论文：** <https://arxiv.org/abs/2611.12177>
- **HTML 原文：** <https://arxiv.org/html/2611.12177v1>
- **项目：** <https://www.embodied-first-exam.ai/>
- **源码：** <https://github.com/mll-lab-nu/embodied-first-exam>
- **核验日期：** 2026-11-11
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

## 对 wiki 的映射

- [Embodied First Exam 主节点](../../wiki/entities/embodied-first-exam.md)
