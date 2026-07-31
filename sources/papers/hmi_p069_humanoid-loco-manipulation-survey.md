# Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning（Humanoid Loco-Manipulation Survey，HMI P069）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning
- **短名：** Humanoid Loco-Manipulation Survey
- **类型：** paper / hmi-papers / LocoManip
- **HMI ID：** P069
- **年份：** 2025
- **原文：** https://arxiv.org/abs/2501.02116
- **代码：** 无 / 见正文开源状态
- **项目页：** 无
- **入库日期：** 2026-07-31
- **一句话说明：** 按控制层、任务类型与真实证据整理人形移动操作研究，统一术语并指出触觉、复杂接触与系统评测缺口。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P069](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P069.md)

## 开源状态（步骤 2.5）

- **结论：** 综述（开源条目随引用工作变化）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

这篇综述的价值不在于给出一个新算法，而是把长期分散在接触规划、运动规划、MPC/WBC、强化学习、模仿学习、基础模型和触觉感知中的工作放到一条系统链上。它还做了一个重要澄清：移动操作关心机器人边移动边操作，whole-body manipulation关心如何把手、胸、腿等所有可用表面变成接触，whole-body loco-manipulation则同时要求两者。

**对 wiki 的映射：** [`wiki/entities/paper-humanoid-loco-manipulation-survey.md`](../../wiki/entities/paper-humanoid-loco-manipulation-survey.md)

### 摘录 2

接触规划决定什么时候用哪只脚、哪只手或身体哪个部位接触什么；运动规划/最优控制在这些模式下求身体、物体和力的轨迹；MPC在短视界内持续重规划；WBC用较高频率将轨迹变成满足动力学、接触锥和关节限制的力矩/位置命令。高保真模型带来约束可解释性，也带来接触组合爆炸和在线计算压力；这就是为什么实际系统常用形心MPC + 局部全身控制的预测-反应层级。

**对 wiki 的映射：** [`wiki/entities/paper-humanoid-loco-manipulation-survey.md`](../../wiki/entities/paper-humanoid-loco-manipulation-survey.md)

### 摘录 3

RL能在仿真里通过大量试错学到扰动恢复和难手工设计的动态行为，但高维、稀疏奖励的loco-manip纯探索成本高；模仿学习用人类或规划器示范缩小搜索空间，但又受重定向、物理可行性和数据覆盖限制。综述明确指出，sim-to-real RL仍依赖仿真动力学模型，它与模型控制不矛盾。更有希望的组合是：规划/约束给结构和安全边界，学习策略处理模型误差、鲁棒性和高维经验。

**对 wiki 的映射：** [`wiki/entities/paper-humanoid-loco-manipulation-survey.md`](../../wiki/entities/paper-humanoid-loco-manipulation-survey.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-humanoid-loco-manipulation-survey.md`](../../wiki/entities/paper-humanoid-loco-manipulation-survey.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)
