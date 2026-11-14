# H-JEPA 官方代码仓库

> 来源归档（ingest · GitHub）

- **类型：** repository
- **仓库：** <https://github.com/kevinghst/H-JEPA>
- **作者 / 维护者：** Wancong Zhang（kevinghst）
- **关联论文：** [H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning](../papers/hjepa_arxiv_2610_06805.md)
- **项目页：** <https://h-jepa.com/>
- **许可证：** MIT（仓库）
- **核查日期：** 2026-10-10
- **沉淀到 wiki：** [H-JEPA](../../wiki/entities/paper-h-jepa-visual-planning.md)

## 仓库内容

官方仓库提供 H-JEPA 的训练与规划代码，面向 Visual AntMaze、FourRoom Distractors、OGBench Cube、Push-T，并支持在 DROID 上进行离线规划。README 提供不同层数（2/3/4 层）的配置和论文结果复现入口；依赖 LeWM 与 stable-worldmodel 相关代码。

## 使用边界

这是研究代码仓库，不代表论文已验证物理机器人闭环部署。复现实验需按 README 准备相应数据集、评测任务及模型检查点，并核对仓库当前环境要求与资源需求。

## 固定版本源码复核（2026-11-13）

- 版本：[97054ecaff83e13839195ec4b92d74c08a0127c5](https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5)。已读根README、eval.py全文，planning_eval.py的load_model/build_solver/_build_policy/run_planning_eval入口和执行路径，hierarchical_solver.py的build_hierarchical_solver/solve/get_cost/_project_to_goal_space完整函数。
- eval.py按clips分派DROID离线或仿真；仿真构建WorldModelPolicy→每层GradientSolver。solve逆层序优化，horizon=1高层可跳过，上一层predicted_embed_0成为下层goal；LevelCostModel把下层rollout投影后计算criterion。最终返回第一级actions。
- README给数据、84仿真+9 DROID模型权重及固定任务下载；提供可运行实现，非网页占位。未下载大数据/权重或运行复现；没有实机闭环验证。DROID分支本次只读文档与入口，未宣称完整核读其评测源码。
