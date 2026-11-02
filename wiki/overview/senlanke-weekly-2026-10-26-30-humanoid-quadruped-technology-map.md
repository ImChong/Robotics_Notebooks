---
type: overview
tags: [survey, humanoid, quadruped, legged-robot, senlanke]
status: complete
updated: 2026-11-02
related:
  - ../tasks/locomotion.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "senlanke 具身运控lab 10-26–10-30 人形/四足周报：33 条论文条目，每条对应唯一 arXiv 详情节点（新建 23 / 复用 10）。"
---

# senlanke 周报索引：10-26–10-30 人形/四足（33 篇）

> **本页定位**：为 [【10.26-10.30前沿论文动态】人形／四足33篇](https://mp.weixin.qq.com/s/eNLudfSs4fI_9YEiBojtJQ) 提供按周报分栏组织的 **33 个条目 → 唯一 arXiv 详情节点**索引。★ 为周报标注的重点论文。

## 一句话观点

本页只按周报原有分栏（人形机器人；四足机器人；轮足与通用腿式机器人）列出条目；论文间的横向优劣不在此判断，请按 arXiv 进各详情节点，条目说明来自周报「问题/创新点」与 arXiv 摘要页。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |
| LiDAR | Light Detection and Ranging | 激光雷达 |
| IMU | Inertial Measurement Unit | 惯性测量单元 |

## 论文索引

### 人形机器人（17）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 1 | [Networked Humanoid Intelligence：网络化人形智能：通信、协调、协作与社会交互](../entities/paper-networked-humanoid-intelligence-survey.md) | [2610.34740](https://arxiv.org/abs/2610.34740) | 新建 |
| 2 | [Zero-HLM：无需真机示范的人形移动操作](../entities/paper-zero-hlm.md) | [2610.35075](https://arxiv.org/abs/2610.35075) | 复用 |
| 3 | [FLoRe（arXiv:2610.35862）](../entities/paper-flore.md) | [2610.35862](https://arxiv.org/abs/2610.35862) | 复用 |
| 4 | [随机动作标记化全身控制：基于随机动作标记化的人形全身控制](../entities/paper-stochastic-motion-tokenization-humanoid-wbc.md) | [2610.36452](https://arxiv.org/abs/2610.36452) | 新建 |
| 5 | [GRAFT：面向基础跟踪器的几何感知残差适配](../entities/paper-graft.md) ★ | [2610.36507](https://arxiv.org/abs/2610.36507) | 复用 |
| 6 | [REACH：面向可适应狭窄空间穿越的分层强化学习人形爬行框架](../entities/paper-reach-humanoid-crawling-confined-space.md) ★ | [2610.36546](https://arxiv.org/abs/2610.36546) | 新建 |
| 7 | [HumiMimic：从自然场景视频迁移人类—人形交互行为](../entities/paper-humimic-human-humanoid-interaction-video.md) | [2610.36615](https://arxiv.org/abs/2610.36615) | 新建 |
| 8 | [HC2Robot：基于超连接 Transformer 与预测前瞻的人形多地形跑酷](../entities/paper-hc2robot-humanoid-parkour.md) ★ | [2610.36825](https://arxiv.org/abs/2610.36825) | 新建 |
| 9 | [ActiveAlign（主动感知）：ActiveAlign：通过主动感知对齐第一视角机器人操作](../entities/paper-activealign-egocentric-active-perception.md) | [2610.36859](https://arxiv.org/abs/2610.36859) | 新建 |
| 10 | [腕目标与关节限位实证：人形手臂重定向中腕部目标构造与关节限位的实证研究](../entities/paper-wrist-target-joint-limit-humanoid-arm-retargeting.md) | [2610.37801](https://arxiv.org/abs/2610.37801) | 新建 |
| 11 | [FORGE：面向敏捷人形运动的在线可行降阶参考生成与全身执行](../entities/paper-forge-humanoid-reduced-order-reference.md) | [2610.37974](https://arxiv.org/abs/2610.37974) | 新建 |
| 12 | [METATRON：可提示的人形全身动作执行层](../entities/paper-metatron-promptable-whole-body-execution.md) | [2610.38027](https://arxiv.org/abs/2610.38027) | 新建 |
| 13 | [Catch It All（arXiv:2610.39044）](../entities/paper-catch-it-all.md) | [2610.39044](https://arxiv.org/abs/2610.39044) | 复用 |
| 14 | [SONIC-K1（arXiv:2610.39327）](../entities/paper-sonic-k1.md) | [2610.39327](https://arxiv.org/abs/2610.39327) | 复用 |
| 15 | [Humanoid Dodgeball：人形躲避球：快速动态障碍规避动作学习](../entities/paper-humanoid-dodgeball-dynamic-obstacle-avoidance.md) | [2610.40003](https://arxiv.org/abs/2610.40003) | 新建 |
| 16 | [DEX-PHD：面向多模态数据采集的便携式灵巧手持设备](../entities/paper-dex-phd-handheld-data-collection.md) | [2610.35462](https://arxiv.org/abs/2610.35462) | 新建 |
| 17 | [连杆驱动人形灵巧手：基于连杆驱动的全关节人形灵巧手](../entities/paper-linkage-driven-humanoid-dexterous-hand.md) | [2610.36789](https://arxiv.org/abs/2610.36789) | 新建 |
### 四足机器人（9）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 1 | [连续时间安全规划控制：面向敏捷四足机器人的连续时间安全规划与控制](../entities/paper-continuous-time-safe-planning-agile-quadrupeds.md) | [2610.35670](https://arxiv.org/abs/2610.35670) | 新建 |
| 2 | [面向四足运动的可训练中央模式发生器框架](../entities/paper-trainable-cpg-quadruped-locomotion.md) | [2610.36537](https://arxiv.org/abs/2610.36537) | 新建 |
| 3 | [相机到四足位姿估计：基于序列图像的外部相机—四足机器人姿态估计](../entities/paper-camera-to-quadruped-pose-estimation.md) | [2610.36599](https://arxiv.org/abs/2610.36599) | 新建 |
| 4 | [黑箱策略传感器诊断修复：黑盒机器人策略失效时：基于透明代理的在线传感器诊断与修复](../entities/paper-blackbox-policy-sensor-diagnosis-repair.md) | [2610.36852](https://arxiv.org/abs/2610.36852) | 新建 |
| 5 | [ELEVATE（arXiv:2610.38388）](../entities/paper-elevate.md) ★ | [2610.38388](https://arxiv.org/abs/2610.38388) | 复用 |
| 6 | [RAIBO-Wagon：四足机器人牵引货车的实时导航系统](../entities/paper-raibo-wagon-legged-towing-navigation.md) | [2610.39267](https://arxiv.org/abs/2610.39267) | 新建 |
| 7 | [LLM 引导平面移动操作：大语言模型引导的四足平面移动操作](../entities/paper-llm-guided-planar-loco-manipulation-quadruped.md) | [2610.40248](https://arxiv.org/abs/2610.40248) | 新建 |
| 8 | [究竟是课程有效，还是起点简单？四足无监督环境设计基准](../entities/paper-unsupervised-environment-design-quadruped-benchmark.md) ★ | [2610.40364](https://arxiv.org/abs/2610.40364) | 新建 |
| 9 | [MUSBRIC：用于可重构肌肉骨骼机器人的乐高兼容气动肌肉模块](../entities/paper-musbric-pneumatic-muscle-bricks.md) | [2610.40285](https://arxiv.org/abs/2610.40285) | 新建 |
### 轮足与通用腿式机器人（7）

| # | 论文节点 | arXiv | 状态 |
|---|----------|-------|------|
| 1 | [Differentiable Contact Legged（arXiv:2610.35852）](../entities/paper-differentiable-contact-legged.md) ★ | [2610.35852](https://arxiv.org/abs/2610.35852) | 复用 |
| 2 | [阶段条件策略优化：多阶段移动操作](../entities/paper-phase-conditioned-loco-manipulation.md) | [2610.36270](https://arxiv.org/abs/2610.36270) | 复用 |
| 3 | [月球单足协同设计：面向月球单足机器人二维跳跃与着陆的双层协同设计](../entities/paper-lunar-monoped-bilevel-codesign.md) | [2610.37199](https://arxiv.org/abs/2610.37199) | 新建 |
| 4 | [RememberWhereYouWalk（arXiv:2610.37203）](../entities/paper-rememberwhereyouwalk.md) | [2610.37203](https://arxiv.org/abs/2610.37203) | 复用 |
| 5 | [TACOS（arXiv:2610.37806）](../entities/paper-tacos.md) ★ | [2610.37806](https://arxiv.org/abs/2610.37806) | 复用 |
| 6 | [Echo（离策略视觉 RL）：Echo：利用离策略经验扩展视觉强化学习](../entities/paper-echo-visual-rl-off-policy-experience.md) ★ | [2610.37865](https://arxiv.org/abs/2610.37865) | 新建 |
| 7 | [OmniQuad：采用全向轮的轮腿混合机器人](../entities/paper-omniquad-wheeled-legged-omnidirectional.md) ★ | [2610.38502](https://arxiv.org/abs/2610.38502) | 新建 |

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [原文链接](https://mp.weixin.qq.com/s/eNLudfSs4fI_9YEiBojtJQ)
