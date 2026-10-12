# AgentGarten: Code Worlds for Evolving Agents（MirroS Blog）

> 来源归档（blog / 官方）

- **标题：** AgentGarten: Code Worlds for Evolving Agents
- **作者：** MirroS Team
- **链接：** <https://mirros.ai/blog/worlds-for-evolving-agents>
- **论文：** <https://arxiv.org/abs/2610.12374>
- **项目页：** <https://mirros-lab.github.io/agent-garten/>
- **代码：** <https://github.com/MirroS-Lab/AgentGarten>
- **入库日期：** 2026-10-12
- **沉淀到 wiki：** [`wiki/entities/paper-agentgarten.md`](../../wiki/entities/paper-agentgarten.md)

## 博客要点

- 每个 hide-and-seek 角色从空 playbook 开始，每轮玩 5 局；行动是短 Python 程序，观察是第一人称渲染帧，无法读取对象坐标、隐藏状态或另一角色的私有观察。
- 轮后 agent 检视获准查看的动作程序与视觉证据，把可迁移技能、失败诊断、后续测试建议写进 playbook；下一轮 agent 从任务说明和历史 playbook 开始。
- 博客将结果与 OpenAI 2019 hide-and-seek 对照：shelter 约 2500 万 RL episode，ramp 约 1 亿；AgentGarten 分别在 round 4 和 round 10 展示。两边的学习先验、观察、实验协议和计数单位不同，适合作历史参照，不能构成控制变量比较。
- 其余四个世界包括陪伴犬、单车道桥、牧羊和采石场装载机；每个任务的分数定义各异。

## 对 wiki 的映射

- [`wiki/entities/paper-agentgarten.md`](../../wiki/entities/paper-agentgarten.md)
- [`sources/papers/agentgarten_arxiv_2610_12374.md`](../papers/agentgarten_arxiv_2610_12374.md)
- [`sources/sites/agentgarten-project.md`](../sites/agentgarten-project.md)
