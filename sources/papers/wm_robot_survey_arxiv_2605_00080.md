# World Model for Robot Learning: A Comprehensive Survey

- **类型**：论文（survey）
- **收录日期**：2026-05-19
- **arXiv**：<https://arxiv.org/abs/2605.00080>（Submitted 30 Apr 2026；v2 18 May 2026）
- **PDF**：<https://arxiv.org/pdf/2605.00080.pdf>
- **项目页**：<https://ntumars.github.io/wm-robot-survey/>
- **代码 / 列表**：<https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy>
- **机构（摘要口径）**：南洋理工大学、UC Berkeley、Stanford、东京大学、Oxford、Microsoft、ETH Zurich、Princeton、Harvard 等

## 一句话

把机器人学习里的 **世界模型** 从「会生成未来视频」拉回 **策略、模拟器与可控视频生成** 三条能力线，强调未来预测必须服务 **学习、评估、规划与执行**，而非停留在开放域视频 demo。

## 为什么值得保留

- **问题重框**：区分通用视频生成、仿真引擎、VLA 后训练与自动驾驶场景预测等同名概念，给出机器人语境下的统一阅读坐标。
- **三线 taxonomy**：（1）**世界模型与策略绑定**（动作前预测环境演化，缓解 VLA 长程误差累积）；（2）**世界模型作为模拟器**（学习式中间训练环境，支撑 RL / 候选动作评估 / 策略验证）；（3）**机器人视频世界模型**（动作/语言/结构条件约束下的可控未来，而非自由续写）。
- **评价口径**：主张下游任务增益、控制一致性与物理一致性优先于「像不像真实视频」的开环指标。

## 核心摘录（面向 wiki 编译）

### 与 VLA / 纯视频生成的分界

- VLA 强在 \(o,l \rightarrow a\) 映射，但长程任务仍面临 **时间信用分配与误差累积**；世界模型补的是 **\(a\) 之后世界如何变** 的预测环节。
- 机器人世界模型的三道门槛（公众号编译稿与综述主线一致）：**物理一致** → **动作可控** → **训练有用**（能提升策略学习或闭环成功率）。

### 机器人视频世界模型的能力拆分（综述图 3 口径）

| 层次 | 解决的问题 |
|------|------------|
| **想象式监督** | 真实交互数据不足时，用预测未来提供动作后果监督（未来不可靠则污染策略） |
| **动作条件** | 未来必须随机器人动作改变，建立因果而非自由续写 |
| **语言条件** | 在任务指令下预测/选择后续状态 |
| **结构条件** | 用深度/三维/物理先验补足像素难以稳定表达的接触与几何 |

### 路线演化（时间线口径）

- 早期：**生成未来观察 → 再反推动作**，动作–结果对齐弱。
- 近期：单一骨干、MoE、统一 VLA、潜空间世界建模等，趋势是 **缩小世界预测与动作决策的距离**，并进入后训练、评估与 RL。

### 开放挑战（综述列题，索引级）

因果条件不足、推理效率、多模态感知（力/触觉）不足、与传统控制结合弱、符号结构不足、**评估指标不成熟**（开环视频指标无法说明策略是否变强）。

## 对 wiki 的映射

- 升格页面：[机器人世界模型：训练闭环与三线 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)
- 交叉补强：[Generative World Models](../../wiki/methods/generative-world-models.md)、[World Action Models（WAM）](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[Model-Based RL](../../wiki/methods/model-based-rl.md)、[人形 RL 运动控制身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)（第 8 层世界模型站位）
- 策展编译：[具身智能研究室公众号解读](../../sources/blogs/wechat_embodied_ai_lab_robot_world_model_training_loop.md)
