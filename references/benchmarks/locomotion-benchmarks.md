# Locomotion 评测

腿足 / 人形运动没有单一的「跑分榜」：不同论文的任务、地形、指标与随机化口径都不一样，跨论文比数字前先确认这几件事是否对齐。

## 常见的评测维度

| 维度 | 典型任务 | 常见指标 |
|------|----------|----------|
| 平地行走 | 速度跟踪、转向 | 跟踪误差、能耗（CoT）、存活时长 |
| 越障 / 复杂地形 | 台阶、碎石、斜坡 | 成功率、通过率、摔倒次数 |
| 高动态 | 跑、跳、跑酷 | 最高速度、腾空时间、落地稳定性 |
| 扰动恢复 | 推力 / 负载扰动 | 可承受冲量、恢复时间 |

## 看数字之前先对齐

- **地形与课程是否同一套**——地形难度分布不同，成功率不可比；
- **随机化范围是否公开**——域随机化开得越大，仿真成绩通常越保守；
- **仿真还是真机**——仿真成绩与真机成绩不能混排；
- 具体怎么挑评测，见 [具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)。

## 关联页面

- [Locomotion 任务页](../../wiki/tasks/locomotion.md)
- [常见失败模式](../../wiki/queries/locomotion-failure-modes.md)
- [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
- [人形训练环境](humanoid-environments.md)
