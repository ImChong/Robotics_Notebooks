# Embodied First Exam — arXiv:2611.12177

- **中文标题：** 具身第一考：机器人环境中的代码智能体能力测评
- **论文：** <https://arxiv.org/abs/2611.12177>
- **HTML 原文：** <https://arxiv.org/html/2611.12177v1>
- **项目：** <https://www.embodied-first-exam.ai/>
- **源码：** <https://github.com/mll-lab-nu/embodied-first-exam>
- **核验日期：** 2026-11-11
- **核对范围：** 论文摘要、标准/特权/实时协议、任务规模与核心成功率、局限段落；尚未逐项通读全部 35 套件说明和所有附录。
- **通读：** 未完成

## 方法与协议

- Coding agent 在机器人环境里编写控制程序，按允许的观测获得反馈，最终由独立 grader 判断是否完成任务。
- **Standard** 为核心感知与动作设置；**Privileged** 允许更丰富状态与重置；**Realtime** 关注在线实时交互。设置不同，结果不应混为一谈。
- 论文统计 **35 套件、1,156 任务、47 种机器人**，论文核心子集 **456 项任务**；最佳受测 agent 在 Standard 核心任务的成功率 **34%**（论文报告）。

## 版本口径警示

2026-11-11 官方仓库 README 已列出 **36 benchmark / 1,211 unique tasks / 2,398 task directories**，核心配置 **498 tasks**；而 README 明确写论文和 leaderboard 统计为 **35 suites / 1,156 tasks / 2,288 tests**（不含后续纳入的 HumanCLAW）。因此不能用新仓库计数替换旧论文结果的分母。旧 wiki 的 32/449/2128 已过时，需在主节点保留论文与当前代码版本双口径。

## 限制

单次尝试评测可能低估多轮试错能力；一些任务没有完整可直接运行的参考解；模拟任务占主导，不能据此推断真实机器人部署成功率。

## 对 wiki 的映射

- [Embodied First Exam 主节点](../../wiki/entities/embodied-first-exam.md)
