# Benchmark 索引 / Benchmarks

这里用于整理 locomotion、humanoid、learning control 等方向最常见的 benchmark 与环境。

`benchmarks/` 的职责是：

> 当你已经知道某个方向大概在研究什么，想进一步知道“大家通常拿什么环境和 benchmark 来比”，从这里进入。

---

## 适合谁看

适合：
- 你已经看过对应 wiki 页面
- 现在想知道某个方向通常怎么 benchmark
- 想搞清楚不同工作常在哪些环境里做比较

不适合：
- 你还没搞懂任务本身是什么
- 你只想看论文列表（去 `papers/`）
- 你只想找 repo（去 `repos/`）

---

## 快速入口

| 你的目标 | 从这里进入 |
|---------|-----------|
| 想看 locomotion benchmark | [Locomotion Benchmarks](locomotion-benchmarks.md) |
| 想看 humanoid 环境 | [Humanoid Environments](humanoid-environments.md) |

---

## 当前主线怎么对应到 benchmarks/

### locomotion 主线
如果你在看：
- Locomotion
- Reinforcement Learning
- Sim2Real
- MuJoCo / Isaac Gym / Isaac Lab

建议从这里进入：
- [Locomotion Benchmarks](locomotion-benchmarks.md)
- [Humanoid Environments](humanoid-environments.md)

### humanoid 研究主线
如果你在看：
- humanoid locomotion
- whole-body control baseline
- RL benchmark

建议从这里进入：
- [Humanoid Environments](humanoid-environments.md)

---

## 当前判断

`benchmarks/` 后续的重点是：

1. 让不同环境的用途更清楚
2. 区分“benchmark 环境”和“训练框架”
3. 让用户更容易判断：某个工作到底在什么设定下比较
