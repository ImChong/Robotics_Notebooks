---
type: method
tags: [rl, sampling, data-generation, humanoid, physics-feasibility, paper, motion-control, body-system-stack, bfm, behavior-foundation-model, nvidia, hku, tsinghua]
status: complete
updated: 2026-11-07
code: https://github.com/OpenDriveLab/AMS
venue: "2025 · arXiv"
arxiv: "2511.17373"
related:
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../overview/bfm-category-02-goal-conditioned-learning.md
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-01-locomotion-base.md
  - ./imitation-learning.md
  - ./beyondmimic.md
  - ../concepts/reward-design.md
  - ../entities/paper-fddc.md
sources:
  - ../../sources/papers/motion_control_projects.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
  - ../../sources/papers/humanoid_rl_stack_18_agility_meets_stability_versatile_humanoid_contr.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/papers/bfm_awesome_ams_arxiv_2511_17373.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/papers/fddc_arxiv_2608_00500.md
summary: "AMS（Agility Meets Stability，arXiv:2511.17373）用经平衡约束过滤的合成平衡动作补充人类动捕数据，并以混合奖励（平衡先验仅用于合成数据）与自适应采样，在单一策略中兼顾敏捷与稳定。"
---

# AMS: 物理可行性过滤与混合奖励

在人形机器人动作合成与学习中，很多参考动捕数据（MoCap）由于人体与机器人的动力学差异，直接模仿会导致物理不可行。**AMS** 提供了一套系统化的数据清洗与奖励设计方法。

## 核心挑战：物理不一致性

人体的质量分布、接触面积和平衡策略与金属构成的机器人截然不同。直接将人类“金鸡独立”或“大跨步”的轨迹映射到机器人上，往往会导致机器人因质心 (CoM) 偏移而不可避免地摔倒。

## 技术方案

### 1. 物理可行性过滤 (Physics Feasibility Filtering)
这一步作用于 AMS **自己合成的平衡动作**（论文算法 1），不是对人类动捕数据逐条清洗。合成参考经两阶段批量轨迹优化（Levenberg–Marquardt 求解）确保“能站稳”：
- **第一阶段：运动学修正**。位姿跟踪、软关节限位、休息姿态正则与时间平滑，确保关节角度满足限位，姿态自然。
- **第二阶段：平衡约束惩罚**。加入惩罚 $\lambda_{bal}\sum\max(0,d_t-\epsilon)$，要求**质心投影 (CoM Projection)** 落在支撑足形成的支撑矩形内。
- **结果**：只有 $\max_t d_t\le\epsilon$ 的合成轨迹被接受，其余丢弃。

### 2. 混合奖励机制 (Hybrid Reward Mechanism)
AMS 提出了一个极具工程参考价值的原则：**不要在简单动作上过度约束**。

- **通用奖励**：涵盖关节误差、末端位姿、速度匹配等基础项。
- **平衡先验奖励**：**按数据来源激活**。
    - 通用跟踪奖励用于**所有**数据；平衡先验奖励（质心对齐、足接触一致性）**只**用于合成平衡动作，人类动捕数据不加（arXiv v3 原文）。论文并没有按“低成功率”或“脚滑严重”去动态开关平衡奖励。
    - 作用：既保留了机器人在人类动捕动作中的运动灵活性，又保证了在极端平衡动作上的稳定性。

### 3. 自适应采样与奖励塑形
- **自适应采样**：按执行失败、平均关节位置误差（MPJPE）与最大关节误差评估每条动作；成功动作以误差 P75 / P25 分位划差 / 好，差者提高采样概率、好者降低，失败动作再额外提升，并设最小采样概率 $p_{min}=\lambda/N$ 防止被完全忽略。
- **自适应奖励塑形**：按动作与身体部位维护独立误差容忍 $\sigma$，以指数滑动平均 $\sigma_{new}=(1-\alpha)\sigma+\alpha\,err$ 更新（承接 PBHC）。
- **数据与平台**：AMASS + LAFAN1 过滤后超过 8,000 条动作，另加 10,000 条合成平衡动作；Isaac Gym 训练，教师（特权）→ 学生蒸馏，真机 Unitree G1（23 自由度）。
- **作者自述局限**：缺乏精确末端执行器控制，难用于操作与接触丰富任务；RGB 位姿估计遥操作的全局运动噪声大，敏捷移动难以遥控。

## 主要技术路线

| 环节 | 实现方案 | 作用 |
|------|---------|------|
| **数据筛选** | 合成动作的质心投影 (CoM) 支撑区约束 | 只接受物理上可站稳的合成平衡动作 |
| **奖励机制** | 混合奖励（平衡先验仅用于合成数据） | 只在合成平衡动作上施加约束，保持灵活性 |
| **管线构建** | 运动学修正 + 物理一致性验证 | 自动生成大规模高质量合成平衡数据 |

## 工程模式：数据生成管线

AMS 实际上定义了一个自动化的合成数据生成流程：
1. **随机采样**：随机采样摆动足位姿、骨盆高度和初始姿态。
2. **轨迹插值**：构造连续的参考运动轨迹。
3. **优化过滤**：通过上述的平衡约束进行物理校验。
4. **高质量池**：形成一个物理可行的、多样化的专家动作数据库。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoCap | Motion Capture | 动作捕捉，参考动作与演示数据的主要来源 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| Reward | Reward Function | 塑造强化学习策略行为的标量反馈 |

## Survey 坐标（策展索引）

### 在 42 篇 RL 运动控制身体系统栈中

| 字段 | 内容 |
|------|------|
| 编号 | 18/42 |
| 系统栈层 | 02 参考跟踪 · 通用控制 |
| 索引来源 | [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) |

### 在 BFM 41 篇技术地图中

| 字段 | 内容 |
|------|------|
| 编号 | 09/41 |
| 分组 | 02 Goal-conditioned 学习 |
| 索引来源 | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) |

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- [AMS 项目主页 (OpenDriveLab)](https://opendrivelab.com/AMS/)
- 原始抓取：[wechat_humanoid_rl_42_survey_2026-05-26.md](../../sources/raw/wechat_humanoid_rl_42_survey_2026-05-26.md)
- 论文：<https://arxiv.org/abs/2511.17373>
- [FDDC（arXiv:2608.00500）](../../sources/papers/fddc_arxiv_2608_00500.md) — 从 AMS 合成平衡集蒸馏 900 条分层单腿 clips，并做 method-agnostic 单腿基准

## 关联页面

- [Reward Design](../concepts/reward-design.md) — AMS 是“条件奖励（Conditional Reward）”设计的典型案例。
- [Imitation Learning](./imitation-learning.md)
- [BeyondMimic](./beyondmimic.md) — BeyondMimic 强调自适应采样；AMS 同样做自适应采样，但特色在合成平衡动作的物理过滤与混合奖励。
- [FDDC](../entities/paper-fddc.md) — 下游单腿基准与可部署动态 CoM；数据子集取自 AMS 合成平衡集
