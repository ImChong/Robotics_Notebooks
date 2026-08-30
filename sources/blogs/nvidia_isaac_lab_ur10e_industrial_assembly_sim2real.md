# Bridging the Sim-to-Real Gap for Industrial Robotic Assembly Applications Using NVIDIA Isaac Lab

> 来源归档（blog / NVIDIA Developer Blog）

- **标题：** Bridging the Sim-to-Real Gap for Industrial Robotic Assembly Applications Using NVIDIA Isaac Lab
- **类型：** blog
- **原始链接：** https://developer.nvidia.com/blog/bridging-the-sim-to-real-gap-for-industrial-robotic-assembly-applications-using-nvidia-isaac-lab/
- **机构：** NVIDIA × Universal Robots（UR10e 真机演示）
- **入库日期：** 2026-08-30
- **一句话说明：** 在 Isaac Lab 中用 IndustReal 思路训练 UR10e **齿轮装配**（抓取 / 自由空间运动 / 插入三技能），经 **Isaac ROS（Segment Anything + FoundationPose）** 感知与 **UR Direct Torque 阻抗控制（500 Hz URScript）** 实现 **零样本 sim-to-real**。

## 开源与项目页核查（步骤 2.5）

| 组件 | 开放程度 | 说明 |
|------|----------|------|
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | **已开源** | Factory 族装配环境（peg / gear mesh / nut thread）已在 Lab 默认任务中 |
| [Isaac ROS](https://github.com/NVIDIA-ISAAC-ROS) | **已开源** | 博客部署链使用 Segment Anything、FoundationPose |
| IndustReal 论文算法 | **已发表** | 博客引用 *IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality* |
| UR10e Direct Torque Command | **早期访问** | UR 提供 early access 力矩接口以支持阻抗控制；非所有 UR 用户默认可用 |
| 本篇齿轮装配训练/部署代码 | **待发布** | 博客文末写「Stay tuned for Isaac Lab environments and training code」——截至入库日 **完整复现包尚未公开** |

## 核心摘录（归纳，非全文）

### 任务分解

- **齿轮装配：** 感知 → 抓取 → 运输 → 插入多轴齿轮到对应轴上
- **三技能：** ① 离线路径 grasp planner；② **motion generation**（RL）；③ **insertion**（RL）
- motion generation 先用 RL 校准训练框架，再攻更难 insertion

### 仿真训练（Isaac Sim 4.5 + Isaac Lab 2.1，RTX 4090）

- **Motion generation MDP：** 随机初始关节角 → 末端到目标位姿；观测含关节位置 + 目标 EE 位姿；动作为关节位置目标；奖励为 EE–目标距离 + 平滑惩罚
- **Insertion MDP：** 齿轮已在夹爪、随机近轴位姿 → 插入轴底；观测含关节位置 + 轴目标位姿；奖励为齿轮–目标距离 + 平滑惩罚
- 策略输出 **60 Hz** 关节位置目标，由仿真内 **低层阻抗控制器** 执行
- **Domain randomization：** 关节摩擦/阻尼、控制器增益、观测噪声；并行多环境 + 随机初始位姿/齿轮尺寸/装配进度
- **网络：** LSTM 256 + MLP [256,128,64]；**PPO（rl-games）**

### 真机部署

- **感知：** RGB → Segment Anything → 分割 mask + 深度 → FoundationPose → 齿轮 6D 位姿
- **观测：** 齿轮位姿 + UR 关节编码器
- **策略输出：** Δ 关节位置 → 绝对关节目标 → **URScript 阻抗控制器 500 Hz** 算力矩
- **结果：** 三颗随机位置齿轮循环装配；对装配顺序与初始位姿鲁棒

### Factory 环境关联

- Isaac Lab 内置 **Factory** 族：`Isaac-Factory-GearMesh-Direct-v0` 等接触密集装配基线（rl_games）；本篇为 **UR10e + IndustReal + Isaac ROS** 的工业落地叙事，不等同于默认 Factory Franka 任务

## 对 wiki 的映射

- [NVIDIA Isaac Lab UR10e 工业装配 Sim2Real](../../wiki/entities/nvidia-isaac-lab-ur10e-industrial-assembly-sim2real.md) — 本篇博客编译页
- [Isaac Lab](../../wiki/entities/isaac-lab.md) — 训练框架与 Factory 环境
- [Isaac Lab 默认环境](../../wiki/entities/isaac-lab-default-environments.md) — Factory / FORGE / AutoMate 装配 ID
- [Sim2Real](../../wiki/concepts/sim2real.md) — 接触密集 manipulation 迁移
- [Manipulation](../../wiki/tasks/manipulation.md) — 装配任务上下文
