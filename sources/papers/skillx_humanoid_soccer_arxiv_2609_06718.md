# SkillX（arXiv:2609.06718）

> 来源归档（ingest）

- **标题：** SkillX：面向人形足球的统一多技能策略学习
- **英文标题：** SkillX: Unified Multi-Skill Policy Learning for Humanoid Soccer
- **类型：** paper / humanoid / soccer / amp / multi-skill / sim2real
- **arXiv：** <https://arxiv.org/abs/2609.06718>（v2 2026-09-11；PDF：<https://arxiv.org/pdf/2609.06718>）
- **Comments：** Accepted to **CoRL 2026**（arXiv 元数据，2026-09-25 核对）
- **项目页：** <https://yzc0731.github.io/SkillX/>
- **机构：** 松延动力（Noetix Robotics）；清华大学
- **作者：** Zhangchen Ye、Enxuan Ruan、Yifei Bao（* 共一）、Runhan Huang、Jiankun Yang、Jiakang Jin、Yixiao Huo、Pengyuan Wang、Yinan Han、Huaxing Huang、Wenhao Cui、Yiming Li（† 通讯）、Xiaoyu Tian（† 通讯）
- **平台：** 25-DoF Noetix E1；Isaac Sim / Isaac Lab 仿真
- **开源：** **待发布**（步骤 2.5 核查 2026-09-25，见 [`sources/sites/skillx.md`](../sites/skillx.md)）
- **入库日期：** 2026-09-14（CoRL 2026 / 作者表 2026-09-25 增量更新）
- **策展索引：** [wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md](../blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 核心论文摘录

### 1) 命令条件统一 actor

- **单一可部署策略** 覆盖盘带、停球、射门等原子技能及长视界组合；技能由命令嵌入选择，推理期保持 **一个 actor**。
- **对 wiki 的映射：** [paper-skillx-humanoid-soccer](../../wiki/entities/paper-skillx-humanoid-soccer.md)

### 2) 技能专属 AMP + critic

- 各技能独立对抗运动先验（保留异构动作风格）与独立价值头（避免多技能价值混淆）；对比共享 AMP / Conditional AMP / MoE-AMP。
- **对 wiki 的映射：** 同上

### 3) 物体感知时序编码器（Transformer + HIM 式辅助目标）

- 聚合观测历史，估计根速度 + **球速度**（仿真特权监督，部署从噪声位置历史推断）；Barlow-Twins 正则。
- **对 wiki 的映射：** 同上

### 4) 仿真组合任务与真机双后端

- Medium：Dribble→Shoot，Hard：Trap→Dribble×3→Shoot；SkillX Overall **88.0% / 81.7%**（1000 trials），最佳基线 MoE-Encoder AMP **67.2% / 45.6%**。
- 真机 MoCap 四任务 10 trial：盘带/射门 **8/10**，两步盘带 **7/10**，盘带+射门 **6/10**；AMP 原子技能仅 **1/10**。
- 机载视觉：ZED2i + VIO + YOLOv8 球检测；另展示去球奖励的泛化交互。
- **对 wiki 的映射：** 同上

## 步骤 2.5 开源核查（2026-09-25）

- 已打开 [项目页](https://yzc0731.github.io/SkillX/)：有 PDF 与演示视频，**无代码链接**（HTML 无 GitHub/HF 入口）。
- arXiv v2 摘要与 HTML 正文 **未列** Code availability URL。
- [Noetix-Robotics/noetix_e1_lab](https://github.com/Noetix-Robotics/noetix_e1_lab) 为 E1 平台 RL 模板，**不能**当作 SkillX 论文实现。
- **结论：** **待发布**；发布后应新建 `sources/repos/skillx.md` 并补 wiki「源码运行时序图」。

## 当前提炼状态

- [x] 项目页步骤 2.5 核查
- [x] wiki 实体页深读更新
- [ ] 官方训练代码（待发布）
