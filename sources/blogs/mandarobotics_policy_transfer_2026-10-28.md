# Do robot policies transfer between physics engines? — 来源归档

## 来源信息

- **发布者**：Manda Robotics
- **标题**：Do robot policies transfer between physics engines?
- **发布日期**：2026-10-28
- **原文**：https://mandarobotics.com/blog/policy-transfer/index.html
- **前篇**：[Comparing physics engines for robotics simulation](mandarobotics_comparing_physics_engines_2026-10-05.md)
- **核对范围**：文章实验方法、结果、局限和附录；页面会更新，以下记录以 2026-11-03 读取内容为准。

## 实验设计

作者使用匹配的资产、场景和策略接口，以 PPO 等策略比较 MuJoCo、Newton 的 MuJoCo-Warp、PhysX 和 Genesis。插入任务在 MuJoCo 训练并在四个实现中评估；ANYmal 行走和 Franka 抓取在 PhysX、Newton 中分别训练并交叉评估；RoboLab 的预训练与 PhysX 微调 π0.5 在两个后端比较。插入实验中的 Newton 使用 MuJoCo-Warp，与 MuJoCo 并非独立接触模型族。

## 主要发现与边界

- **紧间隙插入**：五个 MuJoCo 训练种子各在 50 个相同新初始条件上测评。MuJoCo 成功 50、50、49、50、50 次；PhysX GPU 为 49、47、32、47、11 次，五种子平均 74.4%。该实验把超过接触载荷阈值计为失败。加入力惩罚后 PhysX 平均升至 84.0%，主要来自最弱种子，不能视作普遍有效。
- **求解器设置**：对最脆弱的种子，在 25 个留出初始条件上，PhysX TGS 100/1 为 5/25，PGS 为 21/25，TGS 4/0 为 19/25。更高迭代次数未保证更稳健，也不能由此推断哪个设置更贴近真实。
- **四足**：ANYmal-D 以 1 m/s 前行，PhysX 训练策略在两个后端都使用约高 35% 的关节力矩；五种子、每种子 64 台、10 s 的默认配置下 PhysX→Newton 跌倒 11/320，Newton→PhysX 为 2/320。相同书写的足地摩擦经后端解析后可不同；按关节名称重排观测和动作是迁移前提。该实验主要为探索性比较。
- **抓取和抬升**：标准条件下每个 cell 512 次，Newton→Newton 88%、Newton→PhysX 68%；PhysX→PhysX 82%、PhysX→Newton 53%。每后端仅一个训练种子，初始状态不逐条配对，不能把全部差异归因于动力学，也不能用它单独量化随机化的作用。
- **π0.5 与 RoboLab**：部分场景成功率相近但失败抓取次数不同，另一场景在两个后端的 40 次试验成功数为 17 与 4；原文还检查了场景静止状态与微调。成功率不能替代对接触事件、轨迹与初始化的检查。
- **软球插入附录**：作者将其定位为 pilot，Newton 中有网格无效 episode，保留于统计分母，不能当作收敛或真实物理准确性结论。

## 公开材料与复现状态

[数据与方法](https://mandarobotics.com/blog/policy-transfer/index.html#reproducibility) 及原文末尾链接提供汇总数据、逐次记录、校验和、评测/导出脚本及回放。原文说明原始 checkpoints、冻结场景和完整原生轨迹未包含在下载导出中。页面链接的 Python 脚本在当前读取环境返回不支持的内容类型，尚未完成源码入口和调用链核对；不能据此声称端到端可复现。

## Wiki 映射

- [Manda Robotics 跨引擎比较与策略迁移](../../wiki/entities/mandarobotics-physics-engine-comparison.md)
- [仿真器选型指南](../../wiki/queries/simulator-selection-guide.md)
