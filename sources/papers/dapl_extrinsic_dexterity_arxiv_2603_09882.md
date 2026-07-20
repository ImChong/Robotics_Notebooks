# DAPL: Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning

> 来源归档

- **标题：** Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning
- **类型：** paper
- **出处：** 2026 · RSS 2026 Finalist · arXiv preprint
- **arXiv：** <https://arxiv.org/abs/2603.09882>
- **论文 HTML：** <https://arxiv.org/html/2603.09882>
- **项目页：** <https://pku-epic.github.io/DAPL/>
- **代码：** <https://github.com/SteveOUO/IsaacLab-nonPrehensile>（**In preparation**，截至 2026-07-20 尚未完整发布）
- **作者：** Yixin Zheng, Jiangran Lyu（共同第一）; 通讯 He Wang（Galbot/PKU）；机构：CASIA, BAAI, Galbot, PKU, SJTU
- **入库日期：** 2026-07-20
- **一句话说明：** 世界模型学习杂乱场景接触诱导动力学表示 → RL 以表示引导策略 → 推/拨/翻转外在灵巧性涌现；仿真 >25% 增益，真机 ~50% 成功率（10 场景）；RSS 2026 Finalist。

---

## 核心摘录（策展，非全文）

### 问题与动机

- **杂乱场景取物**：货架等场景中目标物被多个障碍物包围，直接抓取被阻断，需推开/拨动/翻转障碍物。
- 传统规划需精确物体模型；现有学习方法缺乏对接触诱导动力学的显式感知，策略盲目尝试。
- 核心问题：如何让策略「感知」自身动作引发的障碍物动力学变化，从而涌现出有目的的外在灵巧性行为？

### 关键贡献

1. **接触动力学感知框架（DAPL）：** 世界模型从动作-观测历史提炼接触动力学表示 z_t，注入 RL 策略。
2. **外在灵巧性自主涌现：** 不编程原语，仅通过 RL 最大化取物成功奖励，推/拨/翻转行为自然涌现。
3. **仿真 + 真机双验证：** IsaacLab 仿真 >25% 增益；Galbot 等真机 ~50% 成功率（10 场景）。

### 方法要点

| 维度 | DAPL |
|------|------|
| 世界模型输入 | 多帧 (o_t, a_t) 序列 |
| 表示 z_t | 接触诱导场景变化的紧凑编码 |
| 辅助任务 | 障碍物下一状态预测（强迫 z_t 含有接触动力学） |
| RL 训练 | 以 z_t 增强观测；PPO 或 SAC；奖励 = 取物成功 |
| 操作模式 | 推（push）/ 拨（poke）/ 翻转（flip）；非精细抓取 |

### 实验摘要

- **仿真（IsaacLab）：** DAPL vs 基线（无动力学感知）；成功率绝对提升 >25 pp。
- **真机（货架场景）：** 10 种不同杂乱配置；DAPL ~50% 成功率（具体数值见论文 Table）。
- **消融：** 去掉 z_t 注入（无动力学感知）→ 显著下降；去掉辅助任务 → 中等下降。

### 代码状态

- 项目页：<https://pku-epic.github.io/DAPL/>，显示 "Code (In preparation)"。
- 关联仓库：<https://github.com/SteveOUO/IsaacLab-nonPrehensile>，截至核查日内容处于筹备状态。

### 局限（论文自述）

- 真机成功率约 50%，仍存在 sim-to-real gap。
- 依赖 IsaacLab 仿真精度；对极端质量/形状障碍物泛化有限。

### 对 wiki 的映射

- [paper-dapl-extrinsic-dexterity-clutter](../../wiki/entities/paper-dapl-extrinsic-dexterity-clutter.md)
- [manipulation](../../wiki/tasks/manipulation.md)
- [contact-rich-manipulation](../../wiki/concepts/contact-rich-manipulation.md)
- [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2603.09882>
- 项目页：<https://pku-epic.github.io/DAPL/>
- 代码（In preparation）：<https://github.com/SteveOUO/IsaacLab-nonPrehensile>
