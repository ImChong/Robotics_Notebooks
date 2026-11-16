# Gödel Agent — 论文原文归档（arXiv:2410.04444v4）

## 元数据

- **类型：** paper
- **英文题名：** Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement（arXiv 摘要页标题；v4 PDF/ACL 正文标题作 *Recursively Self-Improvement*）
- **中文题名：** Gödel Agent：用于递归自我改进的自指智能体框架
- **作者：** Xunjian Yin, Xinyi Wang, Liangming Pan, Li Lin, Xiaojun Wan, William Yang Wang（按 v4）
- **机构：** 北京大学（Peking University）、加州大学圣塔芭芭拉分校（UCSB）、亚利桑那大学（University of Arizona）
- **论文：** https://arxiv.org/abs/2410.04444v4
- **PDF：** https://arxiv.org/pdf/2410.04444v4
- **HTML：** https://arxiv.org/html/2410.04444v4
- **正式发表：** ACL 2025 Main，https://aclanthology.org/2025.acl-long.1354/
- **代码：** https://github.com/Arvid-pku/Godel_Agent
- **版本：** arXiv v4，2025-05-31；首次提交 2024-10-06
- **核对日期：** 2026-11-08
- **核对范围：** 通读 v4 的摘要、引言、相关工作、§3 算法 1、§4 实现、§5 四基准实验、§6 消融/鲁棒性/Game of 24、§7 讨论/安全、§8 结论与 Limitations/Ethics，以及附录 A–F 的目标提示、实验划分、代表性策略代码、成本与随机采样对比；用官方 src 核对调用路径。
- **通读：** 已完成

## 研究问题与方法

手工设计的 agent 策略固定；传统 meta-learning 的更新算法也固定。Gödel Agent 的自指性指**能够读取并改写自身运行时程序**，包括执行改进操作的程序本身；并不意味着意识或自觉。

论文将固定的更新器 `I` 推广为可自更新的 `I_t`：`(π_{t+1}, I_{t+1}) = I_t(π_t, I_t, r_t, g)`，`r_t=U(E,π_t)`。从初始策略（CoT）、高层目标与任务评价函数出发，由 LLM 生成动作调用；核心操作包括自省、环境交互、更新逻辑、继续递归，实际实现还增加思考、错误处理、代码执行和调用其他 LLM。

运行时通过 Python inspect/动态 exec 与 monkey patching 修改模块函数、类方法；主循环采用递归调用，使下一层使用新逻辑。修改代码可扩充工具和决策流程，不只是改 prompt。

## 实验与事实边界

| 指标 | Meta Agent Search | Gödel-base | Gödel-free |
|---|---:|---:|---:|
| DROP F1 | 79.4 | 80.9 | 90.5 |
| MGSM Accuracy % | 53.4 | 64.2 | 90.6 |
| MMLU Accuracy % | 69.6 | 70.9 | 87.9 |
| GPQA Accuracy % | 34.6 | 34.9 | 55.7 |

- 数据来自论文 v4 Table 1；`Gödel-base` 为受限公平对照，`Gödel-free` 为可使用更强外部模型的无限制设置，**不可把自由设置与基线的差异当作纯算法同资源收益**。
- 每项任务 6 次独立改进，每次最多 30 轮；GPQA 验证 32 / 测试 166，另三项验证 128 / 测试 800；论文附录说明自改生成器使用 GPT-4o-2024-05-13，受评任务策略与基线使用 GPT-3.5-turbo-0125。
- MGSM 消融：去掉 thinking 50.8、去掉错误处理 49.4（完整 Gödel-base 64.2）；错误反馈和先思考再动手有关键作用。
- 100 次 MGSM 试验：4% 意外终止、92% 过程中临时退化、14% 最终未超过初始策略。这些统计**不是**稳健最优的证明。
- Game of 24 的一条示例从 LLM 直接求解换成搜索算法，报告达到 100%；个案不能外推为普遍性能。
- 论文附录 D 报告单次 30 步优化的 API 成本约 $15，对比 Meta Agent Search 约 $300，属于论文实验条件和历史价格，不代表当前实际成本。

## 局限与开放问题

1. 需要具备可靠反馈的任务；基准集中在语言理解、数学与科学 QA，不包含机器人的控制或真机 Sim2Real 评测。
2. 自写自改存在错误、终止、退化与运行时代码执行风险；论文伦理声明建议沙箱和修改边界。
3. 优化器在增长后的自身代码复杂性下可能失去可理解性；并未证明全局最优。
4. 正式论文与当前官方仓库的默认模型/任务配置不同，结果需要独立复现，不能只运行 README 就声称重现 Table 1。

## 对 wiki 的映射

- [Gödel Agent 独立实体](../../wiki/entities/paper-godel-agent.md)
- [递归自改进概念](../../wiki/concepts/recursive-self-improvement.md)

## 一手来源

- https://arxiv.org/html/2410.04444v4
- https://arxiv.org/pdf/2410.04444v4
- https://aclanthology.org/2025.acl-long.1354/
- https://github.com/Arvid-pku/Godel_Agent
