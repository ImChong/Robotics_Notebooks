# Residual Policy / Residual RL 机器人论文精读清单摘录（维护者整理）

- **类型**：`personal`（维护者策展的论文清单与方法归纳，非正式出版物）
- **日期**：2026-07-28
- **用途**：为 [Residual Policy Learning（残差策略学习）](../../wiki/methods/residual-policy-learning.md) 方法页与 9 篇论文实体页提供可追溯的编译来源；各论文细节以对应实体页与 arXiv 原文为准。
- **覆盖范围**：Residual RL / Residual Policy Learning 经典两篇、动作模仿（RFC）、腿足机器人（Versatile Jumping、Multi-Modal ARRL）、技能空间（ReSkill）、共享自治（RSA）、G1 人形近期工作（RuN、ResMimic）。

## 核心形式

Residual Policy 家族的统一形式：最终动作 = 基础动作 + 学习到的补偿量

$$a_t = a_t^{\text{base}} + a_t^{\text{residual}}$$

基础动作可来自传统控制器、MPC、参考轨迹、已有策略甚至人的操作命令；RL 只学习补偿量，从而大幅收窄探索空间、提升样本效率，并让训练初期行为至少不差于 base。

## 论文清单（9 篇）

| # | 论文 | 年份/出处 | 基础部分 | Residual 输出 | 开源状态（入库日核查） |
|---|------|-----------|----------|---------------|------------------------|
| 1 | Residual Reinforcement Learning for Robot Control（Johannink et al.） | ICRA 2019 | 传统反馈控制器（阻抗/位置控制） | 控制量修正（TD3） | 项目页仅论文+视频，**无官方代码** |
| 2 | Residual Policy Learning（Silver, Allen, Tenenbaum, Kaelbling） | arXiv 2018 | 人工控制器 / MPC（已知或学习模型） | Action 修正（DDPG+HER） | **已开源**：k-r-allen/residual-policy-learning |
| 3 | Residual Force Control（Yuan, Kitani） | NeurIPS 2020 | 模仿控制策略（DeepMimic 系） | 作用于根部的外部残差力 | **已开源**（非商用许可）：Khrylx/RFC |
| 4 | Continuous Versatile Jumping Using Learned Action Residuals（Yang et al.） | L4DC 2022（PMLR 211, 2023） | 手工加速度控制器 + WBC | 机身位姿/加速度修正（ARS） | 项目页无代码链接，**未开源** |
| 5 | Multi-Modal Legged Locomotion with Automated Residual RL（Yu, Rosendo） | IEEE RA-L / IROS 2022 | PD 反馈控制器 + 开环步态（黑箱优化器同步调参） | 关节角增量修正（TD3/SAC） | **已开源**：Chenaah/Cheetah-Gym 等三仓 |
| 6 | Residual Skill Policies（Rana et al.） | CoRL 2022 | 技能解码器（VAE latent skill） | 原子动作修正（on-policy RL） | **已开源**（MIT）：krishanrana/reskill |
| 7 | Residual Policy Learning for Shared Autonomy（Schaff, Walter） | ICRA 2020 | **人的操作命令** | 最小干预辅助修正（约束 PPO） | **已开源**：cbschaff/rsa |
| 8 | RuN: Residual Policy for Natural Humanoid Locomotion | arXiv 2025 | Conditional Motion Generator（运动先验） | 关节目标修正（PPO） | 截至入库日**未见官方代码/项目页** |
| 9 | ResMimic（Zhao et al.） | arXiv 2025 | 通用 Motion Tracking（GMT）策略 | 全身动作修正（PPO） | **已开源**：amazon-far/ResMimic（已有完整实体页） |

## 推荐阅读顺序（策展）

1. Residual RL（ICRA 2019）与 Residual Policy Learning（2018）— 基础思想
2. Residual Force Control — 动作模仿中的动力学失配补偿
3. Continuous Versatile Jumping — 真实腿足机器人上的控制器打底 + RL 补偿
4. RuN / ResMimic — G1 人形上的现代形态（Motion Generator / GMT 打底）

## 关键洞见（面向 wiki 编译）

- **Base 不一定要是算法**：经典两篇用传统控制器/MPC，RFC 用模仿策略，RSA 直接把人当作 base policy；Residual 的核心是「已有先验行为的加法修正」，与具体 base 形态无关。
- **Residual 解决的四类失配**：接触/摩擦难建模（Johannink）、部分可观测与传感器噪声（Silver）、人体数据与机器人动力学失配（RFC、RuN）、真实与仿真动力学差（Jumping、ResMimic）。
- **工程三件套反复出现**：残差末层零/小增益初始化（训练初 ≈ base）、critic/value burn-in、残差幅值正则（RSA 将其上升为「最小干预」约束目标）。
- **仿真特权警告**：RFC 的根部外力在真实机器人上不存在，只适合仿真训练/动作生成；与真机可用的动作残差（Jumping、RuN、ResMimic）要区分。
- **与微调的区别**：冻结 base + 加性残差保留了 base 的行为先验（微调会破坏），且残差输入可以包含 base 看不到的信息（ResMimic 的物体状态、RSA 的人动作）。

## 对 wiki 的映射

- 方法枢纽页：[wiki/methods/residual-policy-learning.md](../../wiki/methods/residual-policy-learning.md)
- 论文实体页：
  - [paper-residual-rl-robot-control](../../wiki/entities/paper-residual-rl-robot-control.md)（#1 Johannink, ICRA 2019）
  - [paper-residual-policy-learning](../../wiki/entities/paper-residual-policy-learning.md)（#2 Silver RPL, 2018）
  - [paper-rfc-residual-force-control](../../wiki/entities/paper-rfc-residual-force-control.md)（#3 RFC, NeurIPS 2020）
  - [paper-versatile-jumping-action-residuals](../../wiki/entities/paper-versatile-jumping-action-residuals.md)（#4 L4DC 2022）
  - [paper-multimodal-legged-arrl](../../wiki/entities/paper-multimodal-legged-arrl.md)（#5 RA-L/IROS 2022）
  - [paper-reskill-residual-skill-policies](../../wiki/entities/paper-reskill-residual-skill-policies.md)（#6 CoRL 2022）
  - [paper-residual-policy-shared-autonomy](../../wiki/entities/paper-residual-policy-shared-autonomy.md)（#7 ICRA 2020）
  - [paper-notebook-run-residual-policy-for-natural-humanoid-locomot](../../wiki/entities/paper-notebook-run-residual-policy-for-natural-humanoid-locomot.md)（#8 RuN, 2025；由 planned 索引升格为完整实体页）
  - [paper-resmimic](../../wiki/entities/paper-resmimic.md)（#9 已有完整实体页，本次交叉补强）
- 交叉补强：[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[DeepMimic](../../wiki/methods/deepmimic.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)

## 参考来源（原始）

- Johannink et al., ICRA 2019：<https://arxiv.org/abs/1812.03201>；项目页 <https://residualrl.github.io/>
- Silver et al., 2018：<https://arxiv.org/abs/1812.06298>；项目页 <https://k-r-allen.github.io/residual-policy-learning/>；代码 <https://github.com/k-r-allen/residual-policy-learning>
- Yuan & Kitani, NeurIPS 2020：<https://arxiv.org/abs/2006.07364>；项目页 <https://www.ye-yuan.com/rfc>；代码 <https://github.com/Khrylx/RFC>
- Yang et al., L4DC 2022：<https://proceedings.mlr.press/v211/yang23b.html>；项目页 <https://sites.google.com/view/learning-to-jump>
- Yu & Rosendo, RA-L/IROS 2022：<https://arxiv.org/abs/2202.12033>；项目页 <https://chenaah.github.io/multimodal/>；代码 <https://github.com/Chenaah/Cheetah-Trainer>
- Rana et al., CoRL 2022：<https://arxiv.org/abs/2211.02231>；项目页 <https://krishanrana.github.io/reskill/>；代码 <https://github.com/krishanrana/reskill>
- Schaff & Walter, ICRA 2020：<https://arxiv.org/abs/2004.05097>；项目页 <https://ttic.uchicago.edu/~cbschaff/rsa/>；代码 <https://github.com/cbschaff/rsa>
- Li et al., RuN 2025：<https://arxiv.org/abs/2509.20696>
- Zhao et al., ResMimic 2025：<https://arxiv.org/abs/2510.05070>（归档见 [`sources/papers/resmimic_arxiv_2510_05070.md`](../papers/resmimic_arxiv_2510_05070.md)）

## 当前提炼状态

- [x] 9 篇论文项目页/代码开放状态核查（ingest 步骤 2.5）
- [x] 方法枢纽页与 8 个论文实体页编译（ResMimic 复用已有完整页）
- [x] RuN 由 Paper Notebooks planned 索引升格为完整实体页（原地升级，不新建重复节点）
