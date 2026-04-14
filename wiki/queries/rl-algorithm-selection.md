---
type: query
tags: [rl, locomotion, ppo, sac, td3, algorithm-selection]
status: complete
---

> **Query 产物**：本页由以下问题触发：「在足式/人形机器人任务里，PPO、SAC、TD3 应该怎么选？」
> 综合来源：[Reinforcement Learning](../methods/reinforcement-learning.md)、[Locomotion](../tasks/locomotion.md)、[Sim2Real](../concepts/sim2real.md)、[WBC vs RL](../comparisons/wbc-vs-rl.md)

# RL 算法选型：PPO vs SAC vs TD3（足式/人形机器人）

## 一句话结论

- **先上手、先出稳定 baseline**：优先 PPO。
- **要提高样本效率、可复用离线数据**：优先 SAC。
- **动作平滑要求高、控制维度中等、调参能力强**：可以考虑 TD3。

## 选型总表（工程向）

| 维度 | PPO | SAC | TD3 |
|---|---|---|---|
| 策略类型 | On-policy | Off-policy | Off-policy |
| 稳定性（默认配置） | 高 | 中-高 | 中 |
| 样本效率 | 低 | 高 | 高 |
| 并行仿真友好性 | 很高 | 高 | 高 |
| 对奖励噪声敏感性 | 中 | 中-高 | 高 |
| 在 locomotion 社区成熟度 | 很高 | 高 | 中 |
| 常见失败模式 | 训练慢、数据浪费 | Q-value 过估计/温度不稳 | Q 崩溃、策略抖动 |

## 决策流程（实用版）

1. **你是否有大规模并行仿真资源（Isaac Gym / Isaac Lab）？**
   - 有：PPO 优先做第一条基线。
   - 没有：优先考虑 SAC（样本效率更高）。

2. **你是否希望复用历史 replay 或离线数据？**
   - 是：SAC/TD3 更合适。
   - 否：PPO 仍是最稳妥起点。

3. **你的控制目标是“先走起来”还是“极限性能优化”？**
   - 先走起来：PPO。
   - 追求更高样本效率/性能上限：SAC，然后再做混合或蒸馏。

4. **是否计划快速落地到 sim2real？**
   - 计划落地：优先“PPO + DR + 延迟建模 + 观测噪声注入”这条成熟路线。
   - 若必须压缩真实数据量：增加 SAC 支线做数据效率补充。

## 三种推荐策略模板

### 模板 A：默认工程模板（推荐）

`PPO baseline -> reward/obs/action space 打稳 -> DR -> sim2real`

适用：第一次做 locomotion 项目，目标是最快拿到稳定可复现实验结果。

### 模板 B：数据效率模板

`SAC warmup -> replay 扩展 -> 行为约束/正则 -> 部署前安全过滤`

适用：交互数据昂贵、需要复用历史数据，或硬件资源有限。

### 模板 C：双轨模板（中后期）

`PPO 作为稳定主线 + SAC/TD3 作为性能探索支线`

适用：团队并行推进，既要稳定交付也要探索更高上限。

## 常见误区

- **误区 1：只看 benchmark 分数不看可复现性**
  - 机器人项目里“可复现 + 可诊断”比单次最优分数更重要。
- **误区 2：算法不收敛就直接换算法**
  - 很多时候问题在观测空间、奖励设计、终止条件，不在算法本身。
- **误区 3：忽略部署约束**
  - sim2real 里延迟、噪声、执行器饱和比算法名更决定成败。

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Locomotion](../tasks/locomotion.md)
- [Sim2Real](../concepts/sim2real.md)
- [WBC vs RL](../comparisons/wbc-vs-rl.md)
- [Policy Optimization](../methods/policy-optimization.md)

## 参考来源

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Locomotion](../tasks/locomotion.md)
- [Sim2Real](../concepts/sim2real.md)
- [WBC vs RL](../comparisons/wbc-vs-rl.md)
- [sources/papers/locomotion_rl.md](../../sources/papers/locomotion_rl.md)
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md)
