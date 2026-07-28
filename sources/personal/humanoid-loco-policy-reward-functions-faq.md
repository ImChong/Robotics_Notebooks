# 人形运控常见奖励函数分类 FAQ 摘录（维护者整理）

- **类型**：`personal`（对话/答疑整理，非正式出版物）
- **日期**：2026-07-28
- **用途**：为 [人形机器人运控常见奖励函数分类](../../wiki/concepts/humanoid-policy-reward-functions.md) 提供可追溯的编译来源说明；正文以 wiki 页为准，本文件不重复展开技术细节。
- **综合依据**：本库已 ingest 的 [reward_design.md](../papers/reward_design.md)（Rudin legged_gym / Walk These Ways / EUREKA）、[locomotion_rl.md](../papers/locomotion_rl.md)、[privileged_training.md](../papers/privileged_training.md)（步态条件化奖励），以及 legged_gym / Isaac Lab 系开源训练栈的公开 reward 配置（`cfg.rewards.scales` / reward terms）。

## 对话要点（溯源用）

- 人形/腿式运控 RL 的奖励项可按「替谁说话」切成六类：任务与跟踪、姿态与稳定、步态与接触、能效与平滑、安全与硬件、风格与模仿；总奖励是六类加权和 $r=\sum_i w_i r_i$。
- 奖励与观测互为对偶：观测受「部署可得性」约束（真机拿不到就不能给），奖励只在训练期存在、部署即消失，因此奖励项可以任意使用仿真特权真值（接触力、基座线速度、地形），这正是它与观测设计最大的区别。
- 工程上的关键不是「有哪些项」，而是「权重量级排序」：任务项为 1.0 基准，姿态/步态 0.1–0.5，平滑 1e-3–1e-2，能效 1e-4 量级，安全/终止项 5–100 倍大惩罚；跟踪项用 `exp(-x²/σ)` 核而非线性距离，保证目标附近梯度平滑。
- 人形相对四足多出的关键项：直立（重力投影水平分量）、基座高度、手臂默认位姿、足端朝向；缺失时会出现爬行、半蹲、拖脚等典型失败模式。
