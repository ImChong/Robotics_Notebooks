# WBC vs RL: Whole-Body Control vs Reinforcement Learning

人形机器人运动控制领域最常见的两种路线对比。

## 一句话概括

- **WBC（全身控制）**：基于模型和优化，精确、可解释、依赖精确建模
- **RL（强化学习）**：无模型或基于学习的模型，灵活、可泛化、样本效率低

## 核心差异

| 维度 | WBC | RL |
|------|-----|-----|
| **依赖模型** | 需要精确动力学/运动学模型 | 不需要（model-free），或学一个近似模型 |
| **样本效率** | 高（一次优化） | 低（需要大量交互） |
| **泛化能力** | 受模型精度限制 | 可泛化到新环境 |
| **计算实时性** | QP/NMPC 可实时 | 需要GPU或大量计算 |
| **处理接触切换** | 需要精心设计 | 自然处理 |
| **对不确定性的鲁棒性** | 差（模型误差敏感） | 较强（通过随机化等） |
| **人工介入程度** | 高（需要手动设计约束、任务） | 低（reward 设计相对简单） |
| **理论保证** | 有稳定性/收敛性保证 | 弱 |

## 各自适合的场景

### WBC 更适合
- 已知精确模型的场景（工业机器人）
- 需要精确轨迹跟踪的任务
- 需要硬约束（安全、关节限位）
- 算力受限的嵌入式部署

### RL 更适合
- 模型难以获得的场景（软体、接触丰富的环境）
- 需要泛化到新任务/新环境
- 任务难以手工设计控制策略
- 有足够仿真资源的场景

## 融合趋势

越来越多的工作把两者结合：

- **RL 训练策略，WBC 做底层执行**：如 RL 策略输出高层任务指令，WBC 做全身力矩分配
- **RL 训练 low-level policy，WBC 做 high-level**：如 ASE/CALM 的 LLC+HLC 架构
- **WBC 提供 privileged information 给 RL**：训练时用 WBC 计算的量，推理时去掉

## 结论

不是非此即彼，而是看场景。

入门阶段建议：

- 先学 WBC 打好控制基础（建模、优化、QP）
- 再学 RL 理解学习范式
- 最终目标是能用两者配合解决真实问题

## 关联页面

- [Whole-Body Control](../concepts/whole-body-control.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Sim2Real](../concepts/sim2real.md)

## 推荐继续阅读

- [WBC vs RL 论文导航](../../references/papers/whole-body-control.md)
- [Locomotion RL 论文导航](../../references/papers/locomotion-rl.md)
- [ATOM01-Train](https://github.com/Roboparty/atom01_train)（WBC 工程实践）
- PPO (Schulman 2017)、AMP (Peng 2021)、ASE (Peng 2022)（RL 参考里程碑）
