# ENPIRE: Agentic Robot Policy Self-Improvement in the Real World（arXiv:2606.19980）

> 来源归档（paper）

- **标题：** ENPIRE: Agentic Robot Policy Self-Improvement in the Real World
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.19980>（核对的是 v1，2026-06-18，cs.AI）
- **PDF：** <https://arxiv.org/pdf/2606.19980v1>
- **项目页：** <https://research.nvidia.com/labs/gear/enpire/>
- **代码：** https://github.com/NVlabs/ENPIRE
- **作者 / 机构：** Wenli Xiao、Jia Xie、Tonghe Zhang、Haotian Lin（共同一作）等；NVIDIA、CMU、UC Berkeley；通讯指导 Linxi "Jim" Fan、Yuke Zhu、Guanya Shi
- **入库日期：** 2026-11-12
- **一句话说明：** 面向 coding agent 的真机策略自改进 harness：EN 搭环境（安全约束 / 自动验证 / 自动复位），PI 改进策略，R 并行 rollout，E 多 agent 经 Git 进化选择假设；双臂 YAM 机器人 8 站，真机 4 个任务加 RoboCasa365 仿真。
- **核对范围：** 用 pdftotext 通读 v1 PDF 全文：摘要、第 1–5 节（引言、方法、实验 3.1–3.6、局限、相关工作）、参考文献、附录 A–D（环境创建、机器人系统、消融、RoboCasa 接口与评测协议）的全部正文与图注、表 1–2。图中的柱状 / 曲线数值只取了文字层能抽出的部分（如图 12 的 +10.8 pp、图 13 的 token 数、图 16 的检测计数），未逐图目视核对；摘要里 99% 与正文 "一次 rollout 内最多 8 次重试" 口径的关系见 wiki 页。仓库 https://github.com/NVlabs/ENPIRE 在本环境返回 403，仓库内容未核对，仅确认项目页列出该链接。
- **通读：** 已完成

## 开源状态

- 项目页（2026-11-12 打开）列有 Paper 与 GitHub（NVlabs/ENPIRE）入口；仓库内容未能核对，故「已开源」程度不确认，只记「项目页列出仓库链接」。

## 核心摘录

1. **问题与抽象（摘要、第 1 节）：** 缺的抽象是「复位场景、执行策略、验证结果、改进下一轮」的可重复物理反馈回路；ENPIRE = EN（Environment）+ PI（Policy Improvement）+ R（Rollout）+ E（Evolution）。
2. **两阶段（第 2 节）：** 阶段一由 coding agent 依人类反馈构建环境（硬安全约束、二值奖励验证、自动复位，一次性人力成本，之后作为不可变 Gym API）；阶段二全自主改进策略。
3. **多 agent（第 2.2、B.1 节）：** N 个 agent 对 N 台机器人异步测试 N 个假设，经 Git 分支 / cherry-pick / merge 共享做法；8 站，每站 2×I2RT YAM 6-DoF 臂、RTX 5090 单机。
4. **实验（第 3 节）：** 成功率定义为「一次 rollout 在固定重试次数（8）内完成」；Push-T 启发式学习、pin insertion 梯度学习（要求 50 连续成功）、1/4/8 机器人缩放（Push-T 约 5h 到 2h，pin insertion 1.5h 以上到约 40 分钟）。
5. **资源指标（第 2.2、3.6、4 节）：** MRU、GPU 利用率、MTU；机器人数增加时 MRU 下降，8 agent 时 token 成本超线性。
6. **仿真（第 3.5、图 6、D 节）：** RoboCasa365 上 ENPIRE 对比 GR00T N1.5 与 CaP-X（零样本无 autoresearch），每任务 40 个固定 seed 三元组。

## 对 wiki 的映射

- [paper-enpire](../../wiki/entities/paper-enpire.md)
