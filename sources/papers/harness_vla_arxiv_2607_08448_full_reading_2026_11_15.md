# Harness VLA：原文通读与实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2607.08448v3>
- **版本：** 2026-11-15 获取的 HTML；具体正文、附录和代码范围如下。
- **代码：** <https://github.com/RLinf/RPent>
- **核对日期：** 2026-11-15
- **核对范围：** arXiv:2607.08448v3 正文 §1–5、附录 A–F（文件REPL、原语接口、基准与seed、冻结后端、完整提示listing与调用统计），正文结果表及图注；打开官方项目页，读RPent当前README全文及CLI main的参数、记忆准备、runtime/toolkit初始化与planner.solve路径。未执行仿真、模型或硬件。
- **通读：** 已完成

本次复核单独存档；[历史摘录](harness_vla_arxiv_2607_08448.md)保留原有日期和发布状态，不能代替下述版本及协议核对。

## 原文与实现核对

- 论文v3的few-shot在每任务seed0上构建TSM，评价seed另取；LIBERO400次、Pro800次、RoboCasa target50共340次、RoboTwinC2R250次，不把bootstrap rollout计入测试。
- RoboCasa的总体成绩按18/16/16个任务加权，而非按180/80/80个rollout加权。Codex91.6/56.3/13.8对应任务加权55.4%；聚合时不能简单平均三个split或所有rollout。
- Pro的RATS43.8%仅覆盖六个非Long单元，Harness82.4%覆盖八个；headline+38.6pp不是完全同覆盖比较。同冻结π_RLinf八单元50.0%→82.4%更适合归因harness。
- RoboTwin“zero-shot”指clean TSM向randomized场景转移，不是无记忆；Goal zero-shot才是不取目标TSM/GM。不能合并两种zero-shot定义。
- 原语停止条件只决定局部返回，整任务成功仍由官方predicate判定；参考JSONL保留流程而重新绑定空间参数，不是回放旧坐标。
- [项目页](https://harnessvla.github.io/)仍指向[RPent](https://github.com/RLinf/RPent)。2026-11-15 README已列RoboCasa、RoboTwin、Franka/DualFranka支持，推荐按环境装 `.[libero-pro]` 等extra；CLI要求 `--robot`。WAM/SO-101/YAM仍需逐入口核对，不能从整仓宣传推断全部成熟。
- 当前README额外列92.63% Astra/不同memory批次成绩，与论文v3的82.4%不是同模型/同记忆版本，保留分开。CLI树f2e6b1dc：`main`先 `prepare_run_memory`，再 `robot_spec.init_runtime/get_toolkit` 与 `planner.solve`；公开实现已演进，论文文件REPL与当前工具后端不是逐文件一一对应。

## 对 wiki 的映射

- [Harness VLA 详情](../../wiki/entities/paper-harness-vla.md)
