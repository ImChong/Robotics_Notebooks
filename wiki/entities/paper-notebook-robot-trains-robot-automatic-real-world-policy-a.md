---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2508.12252"
related:
  - ../overview/paper-notebook-category-10-sim-to-real.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robot-trains-robot-automatic-real-world-policy-a.md
summary: "人形机器人「直接在真机上做 RL」一直难落地：怕摔坏、奖励难设计、训练效率低、还得有人全程看着。RTR 的核心点子是再加一台机器人当「老师」——用一台 UR5 机械臂在训练全程托举/保护人形、按课程逐步放手、施加扰动、检测失败、自动复位，把原本需要人手的环节全自动化，从而支持长时间、低人工监督的真机训练。配套提出一套 sim-to-real 流程：先在仿真里训一个把动力学编码进单个隐变量的策略，再在真机上只微调这个隐变量 + 重训 critic，实现高效适配。两个真机任务验证：把行走策略微调到精确速度跟踪、以及从零学会荡秋千式摆动。"
---

# Robot Trains Robot

**Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：10_Sim-to-Real），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形机器人「直接在真机上做 RL」一直难落地：怕摔坏、奖励难设计、训练效率低、还得有人全程看着。RTR 的核心点子是再加一台机器人当「老师」——用一台 UR5 机械臂在训练全程托举/保护人形、按课程逐步放手、施加扰动、检测失败、自动复位，把原本需要人手的环节全自动化，从而支持长时间、低人工监督的真机训练。配套提出一套 sim-to-real 流程：先在仿真里训一个把动力学编码进单个隐变量的策略，再在真机上只微调这个隐变量 + 重训 critic，实现高效适配。两个真机任务验证：把行走策略微调到精确速度跟踪、以及从零学会荡秋千式摆动。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| RTR | Robot Trains Robot | 本文框架：机械臂老师训练人形学生 |
| RL | Reinforcement Learning | 强化学习 |
| F/T Sensor | Force/Torque Sensor | 力/力矩传感器，让机械臂柔顺跟随并感知接触 |
| Dynamics Latent | 动力学隐变量 | 把环境物理编码成的一个紧凑向量，sim-to-real 时只调它 |
| Critic | 价值网络 | Actor-Critic 中估计价值的网络，真机阶段重训 |
| UR5 | Universal Robots 5 | 6 自由度协作机械臂，本文充当「老师」 |

## 为什么重要

- **真机 RL 的安全基础设施**：把「怕摔、要人盯」这个最大门槛交给机械臂老师，真机学习从演示变成可规模化流程
- **课程即硬件**：Z 向逐步放手 = 物理课程，比纯软件课程更直接地控制任务难度
- **隐变量微调思路通用**：「冻策略、只调动力学隐变量」给 sim-to-real 提供轻量在线适配范式，可迁到更大人形
- **自动复位是关键**：真机长训瓶颈往往在复位，自动复位让无人值守训练成为可能

## 解决什么问题

真实世界对人形机器人做 RL 有几座大山：

1. **安全**：人形又高又重，摔一次可能损坏硬件、也中断训练； 2. **奖励与监督**：很多技能在真机上难定义奖励，且失败检测/复位往往靠人； 3. **效率**：从零真机学习样本昂贵，纯靠人盯无法长时间跑。

## 核心机制

1. **「机器人训练机器人」框架**：首次系统地用一台机械臂老师，把人形真机训练的**保护、课程、柔顺引导、扰动、判摔、复位**六件事全部自动化，实现低人工监督的长时间真机 RL。
2. **动力学隐变量微调流程**：仿真编码动力学到单个隐变量、真机只调隐变量 + 重训 critic，让 sim-to-real 适配既高效又稳定。
3. **两类真机任务验证**：既能**微调已有行走策略到精确速度跟踪**，也能**从零学会摆动**这种纯真机技能。
4. **完整开源**：代码（Brax + RSL_RL）、硬件搭建（Toddlerbot + UR5）、复现指引齐全，工程参考价值高。

方法拆解（深读笔记小节）：A. 机械臂老师：六大自动化职能；B. Sim-to-Real：三阶段「动力学隐变量」微调。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 10_Sim-to-Real |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning.html> |
| arXiv | <https://arxiv.org/abs/2508.12252> |
| 机构 | Kaizhe Hu、Haochen Shi、Yao He、Weizhuo Wang、C. Karen Liu、Shuran Song（Stanford） |
| 发表 | 2025-08-17（arXiv v1），**CoRL 2025 录用** |
| 项目主页 | [robot-trains-robot.github.io](https://robot-trains-robot.github.io/) |
| 源码 | [hukz18/Robot-Trains-Robot](https://github.com/hukz18/Robot-Trains-Robot) ✅ |
| 笔记阅读日期 | 2026-06-17 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-10-sim-to-real](../overview/paper-notebook-category-10-sim-to-real.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robot-trains-robot-automatic-real-world-policy-a.md](../../sources/papers/humanoid_pnb_robot-trains-robot-automatic-real-world-policy-a.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning.html>
- 论文：<https://arxiv.org/abs/2508.12252>

## 推荐继续阅读

- [机器人论文阅读笔记：Robot Trains Robot](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning.html)
