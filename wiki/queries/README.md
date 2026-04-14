# wiki/queries/

## 这个目录是什么

`wiki/queries/` 存放 **Query 操作的独立产物**。

根据 Karpathy LLM Wiki 模式：

> "Answers can become new wiki pages, enabling knowledge to compound."

当一次 Query 操作产生了跨多个 wiki 页面的综合分析、对比洞见，或者某个具有独立认知价值的结论，这个产物应写回 wiki 而不是留在聊天记录里。

---

## 什么样的内容放在这里

**放这里的条件（满足任一）：**
- 精读了 2 个以上 wiki 页面，综合出新的连接或对比
- 问题本身有较高的通用性，其他人将来也会问到
- 结论不适合直接归入某个已有类别（concepts/methods/tasks/comparisons）

**不放这里的内容：**
- 只是某个概念的补充 → 直接更新对应的 `wiki/concepts/` 或 `wiki/methods/` 页面
- 系统性的方法对比 → 放 `wiki/comparisons/`
- 新的概念 → 放 `wiki/concepts/`

---

## 页面格式要求

每个 query 产物页面，必须在顶部注明触发来源：

```markdown
> **Query 产物**：本页由以下问题触发：「<问题一句话>」
> 综合来源：[页面A](../concepts/xxx.md)、[页面B](../methods/yyy.md)
```

然后按照正常 wiki 页面标准写内容，包括：
- 核心结论
- 关联页面
- 参考来源

---

## 当前页面列表

（每次新增 query 产物后，在此更新）

| 文件 | 触发问题 | 综合来源 |
|------|---------|---------|
| [mpc-wbc-integration](../concepts/mpc-wbc-integration.md) | MPC 和 WBC 在人形机器人 locomotion 里是怎么配合工作的？ | MPC、WBC、Locomotion、Optimal Control |
| [rl-algorithm-selection](./rl-algorithm-selection.md) | 在足式/人形机器人里，PPO / SAC / TD3 怎么选？ | RL、Policy Optimization、Locomotion、Sim2Real |
| [sim2real-checklist](./sim2real-checklist.md) | 从仿真到真机部署，有哪些必须检查的工程事项？ | Sim2Real、Domain Randomization、SysID、Privileged Training |
| [control-architecture-comparison](./control-architecture-comparison.md) | 人形机器人的主流控制架构有哪些，各有什么优劣？ | WBC vs RL、MPC-WBC、RL、IL、TSID |
| [humanoid-hardware-selection](./humanoid-hardware-selection.md) | 做人形机器人运动控制研究，该选哪个硬件平台？ | Locomotion、Sim2Real、Loco-Manipulation |
| [wbc-implementation-guide](./wbc-implementation-guide.md) | 如何从零搭建一个 WBC 控制器？ | WBC、TSID、HQP、Centroidal Dynamics、Contact Estimation |
| [locomotion-reward-design-guide](./locomotion-reward-design-guide.md) | 怎么设计 locomotion RL 的奖励函数？ | RL、Locomotion、Reward Design、Curriculum Learning |

---

## 和 wiki/concepts/mpc-wbc-integration.md 的说明

该页面是早期的 Query 产物（MPC 与 WBC 如何协同工作），但创建时 queries/ 目录尚未存在，因此留在了 `wiki/concepts/`。
后续新的 query 产物统一放入本目录。
