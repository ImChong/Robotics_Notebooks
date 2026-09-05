# HIL: Hybrid Imitation Learning for Dynamic Athletic Control

> 来源归档（ingest）

- **标题：** HIL: Hybrid Imitation Learning for Dynamic Athletic Control
- **预印本旧题：** HIL: Hybrid Imitation Learning of Diverse Parkour Skills from Videos
- **缩写：** **HIL**（Hybrid Imitation Learning）
- **类型：** paper / physics-based character animation / human-scene interaction
- **期刊：** ACM Transactions on Graphics（TOG）2026
- **arXiv：** <https://arxiv.org/abs/2505.12619>（PDF：<https://arxiv.org/pdf/2505.12619>）
- **TOG PDF：** <https://jiashunwang.github.io/HIL/static/mat/Hybrid_Imitation_Learning_TOG.pdf>
- **项目页：** <https://jiashunwang.github.io/HIL/> — [`sources/sites/hil-project.md`](../sites/hil-project.md)
- **Peng 组索引：** <https://xbpeng.github.io/projects/HIL/index.html>
- **演示视频：** <https://youtu.be/le4248gIMME>
- **作者：** Jiashun Wang, Yifeng Jiang, Haotian Zhang, Chen Tessler, Davis Rempe, Jessica Hodgins, Xue Bin Peng
- **机构：** Carnegie Mellon University；NVIDIA；Simon Fraser University（Peng 双属 NVIDIA / SFU）
- **入库日期：** 2026-06-12
- **再核日期：** 2026-09-05（TOG 定稿 + heading 任务 + G1 非官方扩展仓）
- **一句话说明：** 在并行多任务 RL 中联合 **motion tracking** 与 **AMP 式对抗模仿**，用统一 goal-conditioned 观测（无相位 / 无未来姿态）训练单一物理角色控制器，覆盖 **跑酷障碍穿越** 与 **heading / facing 控制**。

## 开源核查（步骤 2.5，2026-09-05）

| 入口 | 结论 |
|------|------|
| 项目页 [jiashunwang.github.io/HIL](https://jiashunwang.github.io/HIL/) | 摘要、演示分区、BibTeX、TOG PDF；**未列 GitHub / 权重** |
| Peng 组页 [xbpeng.github.io/projects/HIL](https://xbpeng.github.io/projects/HIL/index.html) | 指向 PDF / 项目页 / arXiv；**无代码链** |
| [jiashunwang/Hybrid-Motion-Imitation](https://github.com/jiashunwang/Hybrid-Motion-Imitation) | 一作仓，README 自标 **Unofficial implementation and extension of GfR + HIL**，Apache-2.0；**不是** TOG SMPL 角色动画官方代码。任务是 29-DoF Unitree G1 箱攀 / 搬箱，栈为 Holosoma + Isaac Sim |

**结论：** 官方 TOG 角色动画实现 **确认未开源**。G1 侧存在 **可运行的非官方扩展**（训练 / Viser 评测入口齐），归档见 [`sources/repos/hybrid-motion-imitation.md`](../repos/hybrid-motion-imitation.md)。

## 核心摘录

### 1) 问题与动机

- **Motion tracking** 能精确复现单技能，但难以适应新障碍、编排技能序列。
- **纯 AIL/AMP** 更灵活，但易 **mode collapse**（反复用同一动作、与场景脱节）。
- 跑酷需要：多动态特技串联 + 按场景调整行为；参考数据稀缺且缺对齐场景几何。
- TOG 定稿把框架推广到第二任务：**heading control**（目标前进方向 + 朝向），用 ASE sword-and-shield MoCap（约 7 分钟）验证「同一 hybrid 配方可放大到更大运动库」。

### 2) 混合模仿框架

并行两类任务、等概率采样（先 tracking 预训 4B samples，再两模式各半再训 2B）：

| 模式 | 任务 | 奖励 |
|------|------|------|
| **Motion tracking** | 逐帧跟踪参考片段 | 位姿/速度/根高跟踪 \(r^{track}\) + 能量项 + 共享 **style** \(r^{style}\) |
| **Adversarial / AMP** | 跑酷跟目标点，或 heading/facing | 任务奖励 \(r^{task}\) + style \(r^{style}\) |

- **Style reward**：场景条件判别器 \(D(s_{t-n:t}, c_{t-n:t})\)（跑酷 \(n=10\)），类似 AMP + 梯度惩罚；判别器输入含状态转移与场景点云，判断动作是否既自然又**贴合当前障碍**。heading 任务去掉点云。
- **统一观测**：策略输入角色状态 + goal condition——**不用**相位变量或未来参考姿态。跑酷 goal = agent-centric 最近点云 + 目标位置 \(l_t\)；heading goal = 地面平面单位向量 \((\hat d_t, \hat f_t)\)。
- **架构（跑酷）**：策略为 Transformer + PointNet 点云 token；critic 为 MLP，另吃二进制任务指示 \(k_t\)；判别器为 MLP。
- **角色：** SMPL 物理角色；PD 控制；\(\sigma_\pi=0.055\)；Isaac Gym 4096 并行 × 4×V100；仿真 120 Hz、策略 30 Hz；PPO + GAE。

### 3) 数据与初始化

- **跑酷：** YouTube 19 clip / 30 s / 15 技能；TRAM 姿态重建 + 体朝向 hint 修地面；交互式盒几何标注；MaskedMimic 式物理 tracker 洗碰撞/抖动/滑步后再当 tracking 正样本与 PSI。
- **Heading：** ASE sword-and-shield MoCap，约 7 分钟（进退、转身、持械保持朝向）。
- **PSI（Perturbed State Initialization）：** 参考初态加高斯扰动，促进技能衔接、减轻 mode collapse。
- **Early termination：** tracking 关节偏离参考 >0.5 m；跑酷 AMP 摔倒或偏目标 >2 m；heading 头高 <0.3 m。
- 训练障碍：每课随机抽 5 个障碍、间距 2–3 m；评测对位置/朝向/尺度加噪声。

### 4) 结果与局限

跑酷（障碍位姿/尺度扰动，Table 1）：

| 方法 | Skill Acc ↑ | Track Err ↓ | Task completion ↑ |
|------|-------------|-------------|-------------------|
| Task Reward | 0.00 | 1.82 | 0.81 |
| AMP | 0.06 | 1.49 | 0.11 |
| ASE | 0.03 | 1.63 | 0.00 |
| MaskedMimic | 0.50 | 0.41 | 0.00 |
| Task Reward w/ ws | 0.15 | 0.54 | 0.86 |
| AMP w/ ws | 0.54 | 0.37 | 0.85 |
| **HIL** | **0.66** | **0.31** | 0.74 |

- HIL **不以最高完成率为目标**：纯任务/AMP-ws 完成率更高，但技能覆盖窄或动作不自然。HIL 在技能准确率与 DTW 跟踪误差上最优。
- 消融（Table 2）：去判别器 / 去 PSI / 判别器去场景 / 去 critic \(k_t\) 都会伤技能准确率；去 PSI 完成率掉到 0.52。
- 鲁棒：\(\sigma=0.05\) 完成率 >70%；\(\sigma=0.1\) 仍 >50%；训 5 障、测 20 障（\(\sigma=0.03\)）约 40%。
- 可与 SAMP 坐姿混训（跑酷 + 坐椅子）。
- Heading（Table 3）：HIL Direction 0.94 / Facing **0.97** / Return 227；AMP 方向分略高（0.95）但动作更单；ASE 更自然但任务分低；MaskedMimic 全面崩。

**局限：** 偶发不自然恢复；障碍课默认顺序盒几何；数据外布局/类型适应有限；SMPL 仿真过强可被纯任务基线钻空；论文将真机人形列为未来工作。

## 对 wiki 的映射

- 方法页：[`wiki/methods/hil-hybrid-imitation-learning.md`](../../wiki/methods/hil-hybrid-imitation-learning.md)
- 论文实体：[`wiki/entities/paper-hil-hybrid-imitation-learning.md`](../../wiki/entities/paper-hil-hybrid-imitation-learning.md)
- 交叉更新：
  - [`wiki/methods/amp-reward.md`](../../wiki/methods/amp-reward.md) — 场景条件判别器
  - [`wiki/methods/deepmimic.md`](../../wiki/methods/deepmimic.md) — tracking 分支对照
  - [`wiki/methods/ase.md`](../../wiki/methods/ase.md) — heading 基线与 sword-and-shield 数据
  - [`wiki/methods/mtrg-reference-goal-driven-rl.md`](../../wiki/methods/mtrg-reference-goal-driven-rl.md) — 同作者后人形「参考塑形 + 目标泛化」
  - [`wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md`](../../wiki/comparisons/hil-vs-mtrg-vs-zest-parkour-imitation.md)
  - [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)
  - [`wiki/tasks/humanoid-locomotion.md`](../../wiki/tasks/humanoid-locomotion.md)
  - [`wiki/entities/holosoma.md`](../../wiki/entities/holosoma.md) — 非官方 G1 扩展栈

## 参考来源（原始）

- TOG PDF：<https://jiashunwang.github.io/HIL/static/mat/Hybrid_Imitation_Learning_TOG.pdf>
- 项目页：<https://jiashunwang.github.io/HIL/>
- arXiv：<https://arxiv.org/abs/2505.12619>
- 视频：<https://youtu.be/le4248gIMME>
- G1 非官方扩展：<https://github.com/jiashunwang/Hybrid-Motion-Imitation>
