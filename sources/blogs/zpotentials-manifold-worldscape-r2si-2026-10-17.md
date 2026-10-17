# 深度｜超过 GPT-6、登顶 RoboDojo，Manifold AI 如何让机器人从“一次就会”走向持续自进化？（ZPotentials，2026-10-17）

> 来源归档：媒体文章；文中涉及 R²SI 设计和实验的细节按报道口径转述，不等同于公开论文、代码或官方排行榜核验。

- **发布方：** ZPotentials（搜狐号转载）
- **日期：** 2026-10-17
- **原文：** <https://www.sohu.com/a/1089085076_122063396>
- **项目：** Manifold AI / WorldScape R²SI
- **关联论文：** WorldScape Policy 2.0，arXiv:2607.18840（相关基础工作，不是 R²SI 论文）
- **关联项目页：** <https://manifoldai.cn/>
- **官方评测入口：** <https://robodojo-benchmark.com/>

## 报道要点

- 报道将 R²SI 展开为 **Recursive Robot Self-Improvement**，称其以内外两层 Agent 形成迭代闭环：内层 Robot Agent 根据多视角 RGB、本体状态、任务描述和回合记忆调用 Skill 执行；外层 Recursion Agent 跨轮次复盘、修订 Skill，并通过测试、版本管理后更新内层。
- 报道称调用 reactive model（WorldScape Policy，WSP）也是一种 Skill：模型输出一个动作块后控制权返回 Robot Agent，由新观测决定下一步。对相关任务，外层可把运行轨迹整理为训练数据并更新 WSP。
- **push_T 案例（报道口径）：** 首轮方案耗用 723 步，超过 600 步预算；改进夹持方向、解耦平移与转向、分阶段释放并检查后，报告先以 596 步达到满分，后续同一布局降至 389 步。随后冻结 Skill，在三种随机布局中以 222、312、256 步完成。
- **装箱案例（报道口径）：** 外层根据轨迹构造训练数据并更新 WSP；文章没有提供可下载数据、模型检查点或运行代码。
- 文章称系统在 RoboDojo 榜单超过 GPT-6 Astra、GR00T-N1.7 并排名第一。该排名在本归档中标为**报道主张**，不视为独立验证；核验入口见官方榜单归档。

## 证据边界

文章提供系统结构和案例细节，但不是论文或可运行实现。本文没有找到 R²SI 独立论文、代码仓库、模型权重或公开复现材料。不得把报道中的排名、步数及“自进化”概括当成独立实验结论；后续应以作者原始材料和 RoboDojo 官方榜单记录复核。

## 对应知识页

- [WorldScape R²SI 实体页](../../wiki/entities/manifold-worldscape-r2si.md)
- [WorldScape Policy 2.0](../../wiki/entities/paper-worldscape-policy-2.md)
