---
type: entity
tags: [paper, tactile, contact-rich-manipulation, insertion, action-chunking, imitation-learning, xiaomi-robotics]
status: complete
updated: 2026-10-26
arxiv: "2610.16308"
project: https://pace-insertion.github.io/
related:
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ./paper-foretac.md
  - ./worldarena-2.md
sources:
  - ../../sources/papers/pace_arxiv_2610_16308.md
  - ../../sources/sites/pace-insertion.md
summary: "PACE（小米机器人，arXiv:2610.16308）将触觉流拆成接触状态调制、接触事件修正和未来接触后果预测三条路径；五项真机插入任务平均成功率 90.7%。"
---

# PACE：面向动作块插入策略的时序角色触觉条件化

**PACE**（*Temporal-Role Tactile Conditioning for Chunk-Based Insertion Policies*，arXiv:2610.16308）由北京小米机器人技术有限公司团队提出，研究接触丰富的精密插入。它不是把触觉简单拼接到视觉策略，而是按时间作用拆分触觉信息。

## 方法

| 触觉角色 | 作用 |
|---|---|
| 接触状态（Phase） | 平滑持续接触信号，调制 Transformer 特征，帮助策略判断当前插入阶段 |
| 接触事件（Event） | 检出碰撞、滑移或卡滞等突变，对生成动作块前段进行门控残差修正 |
| 接触后果（Prediction） | 训练时预测未来接触变化，塑造动作表征 |

策略以 ACT 风格的动作块生成器为主干。默认输入包括全局/腕部 RGB、 proprioception 和低维触觉力信号；论文也验证了末端力/力矩、关节电流等替代反馈信号，以及与 Diffusion Policy 主干的兼容性。

## 结果

- 五项真机插入任务平均成功率 **90.7%**，较视觉基线和生成时触觉融合基线高 44 个百分点以上。
- ManipulationNet 六种几何间隙配置平均成功率 **88.7%**；论文报告包含 **0.1 mm** 间隙设置。
- 结果覆盖 peg-in-hole、钥匙、插头、USB 与 Ethernet 等插入任务。

## 阅读时注意

这些数字来自论文所述任务与实验条件，不能直接外推到任意机器人、传感器或装配工位。该工作的关键主张是按时序角色组织触觉，而不是单纯增加触觉输入。

## 来源

- [arXiv:2610.16308](https://arxiv.org/abs/2610.16308)
- [PACE 项目页](https://pace-insertion.github.io/)
