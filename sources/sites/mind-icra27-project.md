# MIND 项目页（mind-icra27.github.io）

> 来源归档

- **标题：** MIND: Instruction-Conditioned Incremental Grounding for Robots
- **类型：** site（项目页，匿名评审）
- **URL：** <https://mind-icra27.github.io/>
- **论文：** <https://arxiv.org/abs/2610.19262>（[sources/papers/mind-incremental-grounding_arxiv_2610_19262.md](../papers/mind-incremental-grounding_arxiv_2610_19262.md)）
- **代码：** <https://anonymous.4open.science/r/mind-icra27>（匿名仓库，[sources/repos/mind-icra27.md](../repos/mind-icra27.md)）
- **入库日期：** 2026-10-17
- **一句话说明：** 增量式房间语义判断维护的项目页，含 Unitree G1 实验。

## 页面结构

| 区块 | 内容要点 |
|------|----------|
| The problem | 世界变化时哪些缓存的房间判断仍成立 |
| How it works | A 指令编译为依赖变量，B 事件足迹，C 求交决定复查房间，D 复用未触及判断并确定性重算其余 |
| Hardware | Unitree G1，四区实验室布局，3/4 场景同布局，26 次计分运行；决策、导航、感知分别计分 |
| 数字 | 240 序列 1,200 决策全部保持，评估 30.6 到 11.2；32 个 AI2-THOR episode，26.00 到 10.47 |

未署名机构；评审期间不放论文 PDF。

## 开源核查结论

截至 2026-10-17：仅匿名评审仓库，页面称含生成器、提示词、manifest、模型配置与结果日志；未见正式 GitHub 链接。

## 对 wiki 的映射

- [wiki/entities/paper-mind-incremental-grounding.md](../../wiki/entities/paper-mind-incremental-grounding.md)
