# EgoHumanoid-V2: Human-to-Humanoid Transfer of Coordinated Whole-Body Skills for Loco-Manipulation

> 来源归档（paper / arXiv v1）

- **作者：** Jin Chen, Yiming Jiang, Chongyang Xu, Modi Shi, Shijia Peng, Li Chen, Tianyu Li, Mu Xu, Yilun Chen, Steven Hoi, Hongyang Li
- **机构：** OpenDriveLab at The University of Hong Kong、Alibaba Group、Shanghai Innovation Institute、Fudan University、Beihang University、Sichuan University、Archon Robotics
- **提交日期：** 2026-09-29（v1）；截至 2026-10-12 收录 arXiv 预印本
- **论文：** <https://arxiv.org/abs/2609.37181>
- **全文：** <https://arxiv.org/html/2609.37181v1>
- **PDF：** <https://arxiv.org/pdf/2609.37181>
- **项目页：** <https://opendrivelab.com/EgoHumanoid-V2/>
- **官方代码：** <https://github.com/OpenDriveLab/EgoHumanoid-V2>
- **源码归档：** [仓库核查](../repos/egohumanoid-v2.md)；[项目页核查](../sites/egohumanoid-v2-opendrivelab.md)
- **沉淀到 wiki：** [EgoHumanoid-V2 实体页](../../wiki/entities/paper-egohumanoid-v2.md)

## 摘要式定位

研究问题是如何把第一视角人类示范中的协调全身移动操作，转化为人形机器人的任务策略监督。此前 ego transfer 工作更关注解耦控制下的移动操作场景泛化，V2 以 SONIC 全身跟踪策略为执行底座，提出粗到细的动作对齐：通过约束 IK 做运动学参考修正，再用学习到的执行响应模型和迭代 rollout 做动力学感知精炼；另以机器人手臂渲染与训练期图像增强缩小视觉具身差异。作者在 Unitree G1 的四项真实任务上训练 π₀.₅ VLA，每任务单独训练，不用该任务的机器人示范。

## 方法与数据要点

- **人体采集：** 颈挂 GoPro 采 egocentric RGB，PICO 采人体全身与手姿；两路同步为 50 Hz。部署侧是头挂 GoPro 的 G1 与 BrainCo Revo2 双手。
- **策略输入/输出：** 观测由适配视觉、43 维 G1 本体状态和任务指令构成；输出是 64 维 SONIC motion token 与左右手各 6 维命令。SONIC 执行全身运动，Revo2 驱动器执行手部动作。
- **动作对齐 H/K/D：** H 是将人体运动输入 SONIC 的仿真初始 rollout；K 以约束离线 IK 缩小手/脚末端目标误差，保持参考变化受限及时间连续；D 用配对 rollout 训练 MLP 响应模型，并以真实仿真执行误差迭代优化 token。优化梯度不经物理模拟器，候选从历史记录中筛选，未必是最后一轮。
- **视觉对齐：** SAM2 分割前臂与腕部追踪器，DiffuEraser 修补背景，再以对齐后的 G1 关节姿态渲染手臂并拟合人类前臂/手腕；训练再用随机裁剪、小旋转和颜色扰动。
- **四项任务：** Close Laptop、Open Fridge、Pull Curtain、Close Drawer；最后一项用右脚尖关抽屉，考验脚端放置和支撑平衡。

## 主要结果

| 指标/条件 | 论文结果 |
|-----------|----------|
| 200 条人类示范/任务，四任务分数均值 | 51.67%；每任务 25 条时为 38.33% |
| 对齐后平均末端位置误差 | H 11.21 cm → H+K 4.45 cm → H+K+Dₛ 3.41 cm |
| 对齐后平均末端朝向误差 | H 21.22° → H+K 13.77° → H+K+Dₛ 6.71°（不含脚端关抽屉） |
| 对齐标签对应的真机策略分数 | H 21.25% → H+K 47.08% → H+K+Dₛ 51.67%（200 条/任务） |
| 人类/机器人示范平均单条劳动时间 | 33.30 / 140.85 人秒；报告人类采集少 76.36% |
| Close Laptop 成本对比 | 100 条人类示范 48.33%，50 条机器人示范 63.33%；人类采集劳动少 31.41% |
| Pull Curtain 成本对比 | 200 条人类示范 53.33%，50 条机器人示范 56.67%；人类采集劳动少 26.63% |

任务分数按每次试验的 3 个顺序子任务累计部分完成度，每种条件 20 次；它不是完整任务成功率。关闭笔记本/拉窗帘才纳入人类与遥操作数据成本对比。图像增强与手臂渲染整体互补，但单任务的最优组合不同。三种在线 IK 实现均出现不稳定 SONIC rollout（包括跌倒），作者未继续评策略。

## 证据边界

- 真实评测是四个物理环境任务、四个独立策略；论文的起始位置/物体变化测试仍属于任务内执行，不构成单策略的未见任务泛化。
- 每个条件 20 次，论文未报告误差条；成本以人员配置、设备设置、校准、排障和场景复位等具体统计口径为准。
- 对齐本身仍有不稳定动作，重复 IK 和仿真 rollout 会限制数据处理吞吐量。
- 当前官方仓库只开放了部分采集与动作对齐链路；视觉对齐、VLA 训练流水线和部署管线仍在 README release checklist 中待发布。详见 [源码核查](../repos/egohumanoid-v2.md)。

## 关联资料

- 项目页：<https://opendrivelab.com/EgoHumanoid-V2/>
- 官方源码：<https://github.com/OpenDriveLab/EgoHumanoid-V2>
- Wiki 实体页：[EgoHumanoid-V2](../../wiki/entities/paper-egohumanoid-v2.md)
- 前作：[EgoHumanoid](https://arxiv.org/abs/2602.10106)
