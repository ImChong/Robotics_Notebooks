# 参考导航 / References

这里不是原始资料堆，也不是知识页正文。

`references/` 的职责是：

> 当你已经在 `wiki/` 或 `roadmap/` 里看懂一个概念，想继续往论文、repo、benchmark、工具生态深挖时，从这里进入。

一句话：
- `sources/` = 原材料输入层
- `wiki/` = 结构化知识层
- `references/` = 继续深挖入口层

---

## 这个目录怎么用

如果你：
- 已经知道一个概念是什么
- 想看这个方向的代表论文
- 想找对应的开源项目、训练框架或 benchmark

那就来 `references/`。

如果你还不知道某个概念本身是什么，请先回 `wiki/`。

---

## 快速入口

| 你的目标 | 从这里进入 |
|---------|-----------|
| 想看 locomotion RL 论文 | [papers/locomotion-rl.md](papers/locomotion-rl.md) |
| 想看模仿学习论文 | [papers/imitation-learning.md](papers/imitation-learning.md) |
| 想看 whole-body control / TSID / WBC 论文 | [papers/whole-body-control.md](papers/whole-body-control.md) |
| 想看 sim2real 论文 | [papers/sim2real.md](papers/sim2real.md) |
| 想找仿真平台 / 训练框架 | [repos/simulation.md](repos/simulation.md) / [repos/rl-frameworks.md](repos/rl-frameworks.md) |
| 想看 humanoid / locomotion benchmark | [benchmarks/locomotion-benchmarks.md](benchmarks/locomotion-benchmarks.md) / [benchmarks/humanoid-environments.md](benchmarks/humanoid-environments.md) |

---

## 三个子目录分别负责什么

### papers/
负责：**按主题组织论文入口**

适合你已经知道自己想看哪个方向，比如：
- locomotion RL
- imitation learning
- whole-body control
- sim2real
- survey papers

入口：
- [papers/README.md](papers/README.md)

### repos/
负责：**按用途组织开源生态和工具链**

适合你想找：
- 仿真平台
- RL 训练框架
- humanoid 开源项目
- retarget 工具
- utilities / 底层库

入口：
- [repos/README.md](repos/README.md)

### benchmarks/
负责：**按任务组织 benchmark 与环境**

适合你想知道：
- 某个方向通常怎么 benchmark
- 常见环境有哪些
- 不同工作通常在什么环境里比较

入口：
- [benchmarks/README.md](benchmarks/README.md)

---

## 当前主线怎么从这里继续往下挖

当前 `Robotics_Notebooks` 的主线是：

```text
LIP / ZMP
  ↓
Floating Base Dynamics
  ↓
Contact Dynamics
  ↓
Capture Point / DCM
  ↓
Centroidal Dynamics
  ↓
Trajectory Optimization / MPC
  ↓
TSID / WBC
  ↓
State Estimation / System Identification / Sim2Real
```

如果你想沿这条主线继续深入：

### 控制 / 优化主链
- [papers/whole-body-control.md](papers/whole-body-control.md)
- [papers/survey-papers.md](papers/survey-papers.md)
- [repos/utilities.md](repos/utilities.md)
- [repos/humanoid-projects.md](repos/humanoid-projects.md)

### 学习 / locomotion 主链
- [papers/locomotion-rl.md](papers/locomotion-rl.md)
- [papers/imitation-learning.md](papers/imitation-learning.md)
- [repos/rl-frameworks.md](repos/rl-frameworks.md)
- [benchmarks/locomotion-benchmarks.md](benchmarks/locomotion-benchmarks.md)

### sim2real / 平台主链
- [papers/sim2real.md](papers/sim2real.md)
- [repos/simulation.md](repos/simulation.md)
- [benchmarks/humanoid-environments.md](benchmarks/humanoid-environments.md)

---

## 和 wiki / sources 的边界

### references 不做什么
- 不重复解释概念是什么
- 不替代 wiki 页面
- 不作为原始资料归档层

### references 真正做什么
- 给出按主题整理的论文入口
- 给出按用途整理的 repo / tool / benchmark 入口
- 成为“理解完概念之后下一步去哪看”的桥

---

## 当前判断

`references/` 现在的结构方向已经对了：
- `papers/`
- `repos/`
- `benchmarks/`

下一步重点不是继续随便加链接，而是：

1. 让更多主线 wiki 页反向链接到这里
2. 让各个 references 页面内部信息量更够用
3. 让 `references/` 本身成为一个真正能导航的层，而不是分目录列表
