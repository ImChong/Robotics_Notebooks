# Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记 + arXiv 全文通读）

- **标题：** Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning.html>
- **分类：** 10_Sim-to-Real
- **arXiv：** <https://arxiv.org/abs/2508.12252>（v1: 2025-08-17，CoRL 2025 录用）
- **项目主页：** <https://robot-trains-robot.github.io/>
- **官方代码：** <https://github.com/hukz18/Robot-Trains-Robot>
- **核对范围：** 全文通读论文文本 §1–§6、Fig. 1–6、Table 1（UR5 + ATI Mini45 + ToddlerBot 硬件系统、三阶段动力学隐变量真机微调、20 min 行走适配与 15 min 荡秋千从零学习、Table 1 定量稳定性评测）。
- **通读：** 已完成
- **入库日期：** 2026-07-10（原文通读复核：2026-11-14）
- **一句话说明：** 针对人形机器人真机 RL 易摔坏、人工护航成本高、难自动复位的问题，RTR 以 6-DoF 机械臂 UR5（配 ATI Mini45 F/T 传感器、4 根弹性吊绳、可编程跑步机与 Mini PC）充当「教师」自动化完成保护、XY 导纳跟随、Z 向物理课程放手（-0.02 m / 5e4 步）、主动扰动与摔倒自动复位，并用「仿真 N=1000 DR 预训练动力学隐变量 -> 真机冻 actor 只调隐变量并重训 critic」实现 20 分钟行走适配及 15 分钟荡秋千从零学习（CoRL 2025）。

## 核心摘录（策展与核对摘要）

- **教师-学生硬件规格**：教师臂为 5 kg 负载 UR5 + ATI Mini45 六维力传感器 + 4 根弹性悬挂绳 + 跑步机；学生平台为 ToddlerBot（0.56 m 高、3.4 kg、30 DoF、BOM 低于 6,000 美元）。
- **行走真机微调（20 min）**：Stage 1 在 N=1000 域随机化环境用 MLP 把物理参数 $\mu^{(i)}$ 编为隐变量 $z$ 并经 FiLM 调制 actor；Stage 2 优化通用隐变量 $\tilde{z}$；Stage 3 真机冻结 actor+FiLM、微调 $z$ 与新 critic；Table 1 测得 0.15 m/s 行走 Roll 0.093、Pitch 0.053、EE Force Z 0.954 N，稳定性优于各对比基线。
- **从零真机学习荡秋千（15 min）**：真机采集 50k 步探索样本 -> 离线 RL 预训练 critic -> 初始化新 actor 联合 PPO 优化，采用 FFT 主频幅值奖励与机械臂相位对齐助推/扰动。

## 对 wiki 的映射

- [paper-notebook-robot-trains-robot-automatic-real-world-policy-a](../../wiki/entities/paper-notebook-robot-trains-robot-automatic-real-world-policy-a.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning/Robot_Trains_Robot_Automatic_Real-World_Policy_Adaptation_and_Learning.html>
- 论文：<https://arxiv.org/abs/2508.12252>
