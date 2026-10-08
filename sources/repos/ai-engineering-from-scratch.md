# AI Engineering from Scratch

> 来源归档

- **标题：** AI Engineering from Scratch — Learn it. Build it. Ship it for others.
- **类型：** repo / 开源课程与 AI 工程参考手册
- **维护者：** Rohit Ghumare 与社区贡献者（按项目 README）
- **代码：** <https://github.com/rohitg00/ai-engineering-from-scratch>
- **官方网站：** <https://aiengineeringfromscratch.com>
- **许可证：** 仓库根目录 LICENSE 为 MIT；README 声明课程免费、无需账号或付费墙。
- **源码状态：** 已开源；课程正文、代码、练习、quiz、skills、prompts、learning paths 与网站实现公开。
- **入库日期：** 2026-10-09
- **一句话说明：** 从数学与机器学习基础一路组织到深度学习、强化学习、LLM、Agent、工具协议和生产部署的多语言 AI 工程课程。
- **沉淀到 wiki：** 是 → [AI Engineering from Scratch 实体页](../../wiki/entities/ai-engineering-from-scratch.md)

---

## README 摘要与课程结构

仓库 README 当前介绍 **20 个阶段、523 节课、约 342 小时**，示例代码语言为 Python、TypeScript、Rust 和 Julia。课程从数学和 ML 基础逐步进入深度学习、计算机视觉、NLP、Transformer、生成式 AI、LLM、Agent、基础设施、安全与综合项目；数量和时长是项目方当前展示的课程规模快照，后续可能变化。

一节课按可运行学习过程组织：先讲问题与概念，再从头实现关键算法，之后对照生产库，最后产出可复用的 prompt、skill、agent 或 MCP server。README 建议学习者记录命令、工作目录、退出码和输出，并在能解释结果后继续。

## 机器人学习者可直接使用的章节

- **Phase 9：Reinforcement Learning** — MDP、动态规划、Monte Carlo、Q-learning、DQN、策略梯度、Actor-Critic、PPO、RLHF、多智能体 RL 与 Sim-to-Real。
- **Sim-to-Real Transfer lesson** — 讲域随机化、系统辨识、域适应与 teacher/student；配套代码是 5×5 GridWorld 中随机“侧滑”概率的玩具示例，比较固定参数策略和域随机化策略，不是 MuJoCo、Isaac 或实体机器人实验。
- **Computer Vision lessons** — 包含 3D Vision / NeRF 和世界模型、视频扩散主题，可作为视觉与生成模型的基础入口。
- **Phase 14：Agent Engineering** 与 **Phase 17：Infrastructure & Production** — 面向 Agent loop、工具、评测、部署与边缘推理，适合补齐 LLM/软件工程相关背景。

## 可运行入口

仓库 README 提供一个不需下载模型或 GPU 的入门命令：

~~~bash
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch
python3 phases/00-setup-and-tooling/01-dev-environment/code/verify.py --route beginner
~~~

机器人 RL 示例从仓库根目录运行：

~~~bash
python3 phases/09-reinforcement-learning/11-sim-to-real-transfer/code/main.py
~~~

该脚本用标准库实现 tabular Q-learning，在训练阶段比较固定 slip 与每回合随机 slip，再在训练范围内外测试贪心策略；它用于讲清域随机化的直觉，不提供机器人动力学、硬件接口或可复现的 sim-to-real benchmark。

## 阅读建议

已熟悉机器人 RL 的读者不必按数百小时线性重学。可把它当作按主题索引的补课手册：按需读概率 / ML 先修、PPO、Sim-to-Real 和 Agent / edge inference 章节；将类比代码迁移到机器人前，应回到具体论文、仿真器与硬件文档验证方法和参数。

## 对 wiki 的映射

- [AI Engineering from Scratch 实体页](../../wiki/entities/ai-engineering-from-scratch.md)
- [官方网站归档](../sites/ai-engineering-from-scratch.md)
- [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
- [Sim2Real](../../wiki/concepts/sim2real.md)
- [Agentic Coding 软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)
