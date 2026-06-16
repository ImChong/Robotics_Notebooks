#!/usr/bin/env python3
"""Batch-insert ## 英文缩写速查 for top-50 hub pages (skip if present)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GLOSSARIES: dict[str, str] = {
    "wiki/concepts/behavior-foundation-model.md": """| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| WBC | Whole-Body Control | 人形多关节协调控制，BFM 主要服务层 |
| VLA | Vision-Language-Action | 高层语义策略，常与 BFM 低层执行叠加 |
| GC | Goal-conditioned Learning | 以外在目标/参考条件化全身技能扩展 |
| FB | Forward–Backward Representation | 无 reward 的转移表征，如 BFM-Zero 线 |
| DIAYN | Diversity Is All You Need | 内在奖励技能发现类预训练代表 |""",
    "wiki/entities/humanoid-robot.md": """| DoF | Degrees of Freedom | 人形通常 20–50+ 关节自由度 |
| MoCap | Motion Capture | 人体演示数据来源，支撑 IL/重定向 |
| IL | Imitation Learning | 从专家演示/动捕学习策略 |
| VLA | Vision-Language-Action | 类人形态便于接入多模态基础策略 |
| WBC | Whole-Body Control | 协调多肢满足平衡与操作任务 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机的核心工程议题 |
| CAD | Computer-Aided Design | 机构建模与硬件设计 |
| URDF | Unified Robot Description Format | 机器人动力学模型与仿真导入 |
| MPC | Model Predictive Control | 滚动优化质心/接触的站立步态基线 |
| RL | Reinforcement Learning | 仿真中训练行走与全身策略 |
| HIL | Hardware-in-the-Loop | 真机联仿/台架测试，上保护绳前的安全验证 |
| QDD | Quasi-Direct Drive | 低减速比准直驱，Unitree 等平台常见 |
| SEA | Series Elastic Actuator | 串联弹性执行器，Digit 等平台采用 |
| PRS | Planetary Roller Screw | 行星滚柱丝杠直线腿驱动（如 Optimus） |
| EtherCAT | Ethernet for Control Automation Technology | 工业实时总线，OpenLoong 等全尺寸人形采用 |
| ROS 2 | Robot Operating System 2 | 人形高频控制的通信与中间件栈 |
| FastDDS | Fast Data Distribution Service | ROS 2 默认 DDS 实现，影响控制延迟 |
| HAL | Hardware Abstraction Layer | 屏蔽不同机型的底层驱动差异 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富动力学仿真与分析 |""",
    "wiki/methods/generative-world-models.md": """| WM | World Model | 预测环境动态，供规划/RL/评估使用 |
| GWM | Generative World Model | 用生成式 AI 从视频学习世界规律 |
| RL | Reinforcement Learning | 可在想象 rollout 中试错的训练范式 |
| MBRL | Model-Based Reinforcement Learning | 显式或学习式环境模型的 RL |
| VLA | Vision-Language-Action | 可与世界模型级联或联合训练 |""",
    "wiki/entities/unitree-g1.md": """| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| WBC | Whole-Body Control | 全关节力控，适配全身协调控制 |
| RL | Reinforcement Learning | 常见在 Isaac Lab / legged_gym 训练 |
| LiDAR | Light Detection and Ranging | 机载 3D 激光，支撑地形感知 |
| SD-AMP | Selective Domain AMP | 走跑起身统一策略的代表工作线 |
| PILOT | Perceptive Loco-Manipulation | 感知移动操作 LLC 在 G1 上的验证 |""",
    "wiki/tasks/loco-manipulation.md": """| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 统一分配行走与上肢任务的协调层 |
| VLA | Vision-Language-Action | 高层语义/任务接口，低层全身执行 |
| MPC | Model Predictive Control | 滚动优化质心/接触的经典分层路线 |
| HLC | High-Level Control | 给出末端或技能目标的上层模块 |
| LLC | Low-Level Control | 跟踪全身参考或力矩的底层策略 |""",
    "wiki/concepts/motion-retargeting.md": """| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |
| MoCap | Motion Capture | 最常见参考动作来源 |
| IK | Inverse Kinematics | 满足末端/姿态约束的关节解算 |
| WBT | Whole-Body Tracking | 重定向后用于仿真跟踪训练 |
| AMP | Adversarial Motion Prior | 可与重定向数据组合约束运动风格 |""",
    "wiki/entities/isaac-gym-isaac-lab.md": """| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真（PhysX）训练环境 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习框架 |
| RL | Reinforcement Learning | 大规模并行 PPO 等训练的主场景 |
| GPU | Graphics Processing Unit | 数千环境并行仿真的算力基础 |
| URDF | Unified Robot Description Format | 机器人模型导入仿真的标准描述 |""",
    "wiki/entities/mujoco.md": """| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富刚体仿真引擎 |
| MJCF | MuJoCo XML Format | 模型与场景描述格式 |
| MJX | MuJoCo JAX | JAX/XLA 后端，便于可微与批量 |
| RL | Reinforcement Learning | 腿足/人形 loco 常用训练后端 |
| PD | Proportional–Derivative | 仿真中常见的低层关节控制接口 |""",
    "wiki/overview/robot-world-models-training-loop-taxonomy.md": """| WM | World Model | 预测未来状态/观测以支撑决策 |
| WAM | World Action Model | 联合建模世界动态与动作 |
| RL | Reinforcement Learning | 可在想象环境中 rollouts 微调 |
| VLA | Vision-Language-Action | 级联路线中动作解码的上层 |
| IDM | Inverse Dynamics Model | 由预测未来反推动作的常见模块 |""",
    "wiki/concepts/foundation-policy.md": """| FP | Foundation Policy | 跨任务/跨场景可复用的通用策略抽象 |
| VLA | Vision-Language-Action | 操作域最典型的 foundation policy 实例 |
| IL | Imitation Learning | 大规模演示预训练的主要路线之一 |
| RL | Reinforcement Learning | 与 IL 组合或后训练提升鲁棒性 |
| BC | Behavior Cloning | 监督式模仿的基础范式 |""",
    "wiki/methods/sonic-motion-tracking.md": """| SONIC | Supersizing Motion Tracking for Natural Humanoid WBC | 规模化运动跟踪预训练人形控制 |
| MoCap | Motion Capture | 海量参考轨迹的监督来源 |
| WBC | Whole-Body Control | 全身协调跟踪的执行层问题 |
| VLA | Vision-Language-Action | 经统一 token 接入的高层接口 |
| VR | Virtual Reality | 遥操作与实时参考生成入口之一 |""",
    "wiki/overview/bfm-category-02-goal-conditioned-learning.md": """| BFM | Behavior Foundation Model | 本分类所属运控基座范式 |
| GC | Goal-conditioned Learning | 以目标/参考条件扩展动作覆盖面 |
| HOI | Human–Object Interaction | 人-物交互技能纳入统一目标条件 |
| WBC | Whole-Body Control | 跟踪与全身技能的控制载体 |
| TWIST | Teleoperated Whole-body Imitation System | 遥操作数据采集代表工作 |""",
    "wiki/overview/world-models-15-open-source-technology-map.md": """| WM | World Model | 预测未来以支撑策略学习与评估 |
| WAM | World Action Model | 想象与动作同骨干的联合架构 |
| RL | Reinforcement Learning | 虚拟沙盒路线中的想象环境 |
| VLA | Vision-Language-Action | 级联路线常见上层语义接口 |
| IDM | Inverse Dynamics Model | 由未来表征解码动作块的模块 |""",
    "wiki/queries/humanoid-motion-tracking-method-selection.md": """| WBT | Whole-Body Tracking | 参考动作跟踪类方法总称 |
| AMP | Adversarial Motion Prior | 分布约束式运动先验路线 |
| RL | Reinforcement Learning | 任务奖励与先验联合优化 |
| MoCap | Motion Capture | 参考动作与风格数据来源 |
| Sim2Real | Simulation to Real | 跟踪策略上真机的迁移考量 |""",
    "wiki/methods/amp-reward.md": """| AMP | Adversarial Motion Prior | 判别器约束状态转移接近专家分布 |
| GAN | Generative Adversarial Network | AMP 对抗训练的范式来源 |
| RL | Reinforcement Learning | 任务 reward 与风格 reward 联合优化 |
| ADD | Adversarial Differential Discriminator | 差分判别、减碎片 reward 的演进 |
| HOI | Human–Object Interaction | HumanX 扩展的接触图交互场景 |""",
    "wiki/overview/navigation-slam-autonomy-stack.md": """| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| ROS 2 | Robot Operating System 2 | 导航与系统集成常用中间件 |
| LiDAR | Light Detection and Ranging | 2D/3D 激光 SLAM 主传感器 |
| MPC | Model Predictive Control | 局部路径/轨迹跟踪优化 |
| VLA | Vision-Language-Action | 语义导航与端到端驾驶新路线 |""",
    "wiki/concepts/world-action-models.md": """| WAM | World Action Model | 联合预测世界动态与动作的多模态模型 |
| VLA | Vision-Language-Action | 传统分模块的级联对照基线 |
| WM | World Model | 侧重环境预测、动作后解码的架构 |
| IDM | Inverse Dynamics Model | 由未来潜变量反推动作的常见头 |
| RL | Reinforcement Learning | 可用 WAM 想象 rollout 微调策略 |""",
    "wiki/concepts/contact-rich-manipulation.md": """| CRM | Contact-Rich Manipulation | 多指/多接触约束下的操作任务 |
| WBC | Whole-Body Control | 移动操作中协调力与运动的全身层 |
| IL | Imitation Learning | 接触策略常从示教或扩散策略学习 |
| RL | Reinforcement Learning | 探索接触模式与力控制的路线 |
| 6DoF | Six Degrees of Freedom | 物体位姿级抓取/操作表示 |""",
    "wiki/concepts/whole-body-tracking-pipeline.md": """| WBT | Whole-Body Tracking Pipeline | 参考→仿真跟踪→部署的流水线 |
| MoCap | Motion Capture | 上游人体/专家参考来源 |
| RL | Reinforcement Learning | 仿真中 PPO 等学跟踪策略 |
| Sim2Real | Simulation to Real | 跟踪策略迁移真机阶段 |
| Retargeting | Motion Retargeting | 流水线首段：映射到机器人骨架 |""",
    "wiki/methods/motion-retargeting-gmr.md": """| GMR | General Motion Retargeting | 通用人体→机器人动作重定向方法 |
| MoCap | Motion Capture | 输入运动序列的主要来源 |
| IK | Inverse Kinematics | 满足关节限位与末端约束的求解 |
| SMPL | Skinned Multi-Person Linear Model | 常见人体参数化与重定向源 |
| WBT | Whole-Body Tracking | 重定向产物用于下游跟踪训练 |""",
    "wiki/methods/model-predictive-control.md": """| MPC | Model Predictive Control | 滚动时域内优化控制序列 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制 |
| QP | Quadratic Programming | 线性化动力学下的常见求解形式 |
| CoM | Center of Mass | 质心轨迹是 loco MPC 核心状态 |
| WBC | Whole-Body Control | MPC 输出参考，低层 QP 跟踪全身 |""",
    "wiki/concepts/motion-retargeting-pipeline.md": """| Retargeting | Motion Retargeting | 人体参考→机器人可执行动作 |
| GMR | General Motion Retargeting | 本库常用重定向工具链节点 |
| WBT | Whole-Body Tracking | 重定向后仿真跟踪训练段 |
| MoCap | Motion Capture | 流水线输入的动捕/视频数据 |
| Sim2Real | Simulation to Real | 跟踪策略上真机的最后一段 |""",
    "wiki/entities/legged-gym.md": """| legged_gym | Legged Gym | ETH RSL 四足/双足 RL 训练框架 |
| RL | Reinforcement Learning | PPO 训 loco 的主流入口 |
| PD | Proportional–Derivative | 策略输出经 PD 转力矩执行 |
| GPU | Graphics Processing Unit | Isaac Gym 并行仿真依赖 |
| Sim2Real | Simulation to Real | 训练后零样本/微调上真机 |""",
    "wiki/tasks/teleoperation.md": """| Teleop | Teleoperation | 人远程操控机器人采集演示 |
| IL | Imitation Learning | 遥操作数据常用于 BC/扩散策略 |
| VR | Virtual Reality | 全身遥操作与参考生成接口 |
| MoCap | Motion Capture | 与遥操作并列的高质量动作来源 |
| WBC | Whole-Body Control | 低层执行全身跟踪或力控 |""",
    "wiki/overview/ego-9-papers-technology-map.md": """| Ego | Egocentric Vision | 第一人称视角感知与控制的论文簇 |
| VLA | Vision-Language-Action | 若干工作把 ego 视频接入策略 |
| WM | World Model | 自我中心未来预测与规划 |
| SLAM | Simultaneous Localization and Mapping | 导航类 ego 工作相关背景 |
| RL | Reinforcement Learning | 部分 ego 控制采用端到端 RL |""",
    "wiki/queries/legged-humanoid-rl-pd-gain-setting.md": """| PD | Proportional–Derivative | 关节刚度/阻尼底层，RL 输出常为其 setpoint |
| Kp | Proportional Gain | 位置误差增益，影响刚度与响应 |
| Kd | Derivative Gain | 速度误差增益，抑制振荡 |
| RL | Reinforcement Learning | 策略层与 PD 层分工是 loco 常见栈 |
| Sim2Real | Simulation to Real | 增益不匹配是迁移失败常见原因 |""",
    "wiki/methods/model-based-rl.md": """| MBRL | Model-Based Reinforcement Learning | 先学/用环境模型再规划或想象 rollout |
| RL | Reinforcement Learning | 与 model-free 对照的总称 |
| MDP | Markov Decision Process | 状态–动作–转移的标准建模 |
| MPC | Model Predictive Control | 与 learned model 结合的规划实例 |
| Dreamer | Dreamer (World Models) | 潜空间想象训练的 MBRL 代表 |""",
    "wiki/tasks/stair-obstacle-perceptive-locomotion.md": """| Locomotion | Robot Locomotion | 楼梯/障碍场景下的足式移动任务 |
| DCM | Divergent Component of Motion | 落脚点规划与 capture 相关概念 |
| RL | Reinforcement Learning | PPO 等学感知行走策略 |
| PPO | Proximal Policy Optimization | 显式几何/感知条件化 loco 常用算法 |
| Sim2Real | Simulation to Real | 感知策略从仿真到户外真机 |""",
    "wiki/concepts/domain-randomization.md": """| DR | Domain Randomization | 训练时随机化仿真参数以提升鲁棒迁移 |
| Sim2Real | Simulation to Real | DR 是缩小 domain gap 的核心手段之一 |
| RL | Reinforcement Learning | DR 多在仿真 RL 训练阶段施加 |
| SysID | System Identification | 与 DR 互补：缩小参数而非覆盖分布 |
| URDF | Unified Robot Description Format | 被随机化的质量/摩擦等常来自模型参数 |""",
    "wiki/overview/robot-learning-overview.md": """| RL | Reinforcement Learning | 从交互奖励学习策略 |
| IL | Imitation Learning | 从专家演示学习策略 |
| Sim2Real | Simulation to Real | 学习策略落地真机的工程主线 |
| VLA | Vision-Language-Action | 多模态基础策略方向 |
| WBC | Whole-Body Control | 人形/移动操作的控制基础设施 |""",
    "wiki/methods/diffusion-policy.md": """| DP | Diffusion Policy | 用扩散模型生成动作序列的 IL 方法 |
| IL | Imitation Learning | 从示教学习，DP 是其生成式分支 |
| BC | Behavior Cloning | 确定性回归对照基线 |
| ACT | Action Chunking Transformer | 常与 DP 并列的序列动作预测架构 |
| Sim2Real | Simulation to Real | 操作策略迁移真机的后续阶段 |""",
    "wiki/entities/paper-behavior-foundation-model-humanoid.md": """| BFM | Behavior Foundation Model for Humanoid Robots | 本文提出的人形全身行为基础模型 |
| WBC | Whole-Body Control | 预训练服务低层全身协调 |
| RL | Reinforcement Learning | 预训练与下游适应常用范式 |
| MoCap | Motion Capture | 大规模行为数据主要来源之一 |
| VLA | Vision-Language-Action | 可与 BFM 低层能力上下叠加 |""",
    "wiki/entities/open-source-humanoid-hardware.md": """| DoF | Degrees of Freedom | 各平台关节规模对比维度 |
| BOM | Bill of Materials | 自研/开源硬件物料清单 |
| RL | Reinforcement Learning | 平台选型常看仿真与训练生态 |
| WBC | Whole-Body Control | 力控/全身协调硬件能力指标 |
| Sim2Real | Simulation to Real | 硬件–仿真模型一致性影响迁移 |""",
    "wiki/methods/policy-optimization.md": """| PO | Policy Optimization | 直接优化策略参数的 RL 算法族 |
| PPO | Proximal Policy Optimization | 机器人 loco 最常用 on-policy 算法 |
| TRPO | Trust Region Policy Optimization | 信任域约束的策略梯度先驱 |
| RL | Reinforcement Learning | 策略优化是 model-free RL 核心 |
| MDP | Markov Decision Process | 策略优化的标准问题形式 |""",
    "wiki/concepts/reward-design.md": """| Reward | Reward Function | 标量反馈，塑造 RL 策略行为 |
| RL | Reinforcement Learning | reward 设计是 sample efficiency 关键 |
| AMP | Adversarial Motion Prior | 风格/分布类 reward 与任务 reward 组合 |
| PPO | Proximal Policy Optimization | 对 reward 尺度与稀疏性较敏感 |
| WBC | Whole-Body Control | 任务空间 reward 常需与全身约束一致 |""",
    "wiki/concepts/privileged-training.md": """| Privileged Info | Privileged Information | 训练时可访问、部署时不可见的额外状态 |
| Teacher–Student | Teacher–Student Distillation | 特权教师策略蒸馏可部署学生 |
| RMA | Rapid Motor Adaptation | 从历史隐式估计环境参数的实例 |
| RL | Reinforcement Learning | 特权信息常在仿真训练阶段使用 |
| Sim2Real | Simulation to Real | 特权训练是缩小 gap 的常见技巧 |""",
    "wiki/methods/beyondmimic.md": """| BeyondMimic | BeyondMimic Framework | 高精度仿真人形动作模仿框架 |
| IL | Imitation Learning | 参考轨迹跟踪式模仿学习 |
| RL | Reinforcement Learning | 仿真中 PPO 等优化跟踪策略 |
| Isaac Lab | NVIDIA Isaac Lab | 主要验证与训练环境 |
| Sim2Real | Simulation to Real | 强调物理建模与采样以促迁移 |""",
    "wiki/overview/bfm-category-05-hierarchical-control.md": """| BFM | Behavior Foundation Model | 层次化控制中的低层身体执行 |
| HRL | Hierarchical Reinforcement Learning | 高层技能 + 低层控制的结构 |
| LLM | Large Language Model | 部分工作作高层任务/语言接口 |
| VLA | Vision-Language-Action | 与 BFM 低层组合的上层策略 |
| WBC | Whole-Body Control | 低层跟踪与全身协调 |""",
    "wiki/entities/quadruped-robot.md": """| Quadruped | Quadruped Robot | 四足平台，静态稳定裕度通常大于双足 |
| RL | Reinforcement Learning | 四足 loco 并行仿真训练成熟 |
| PD | Proportional–Derivative | 策略输出经 PD 转关节力矩 |
| Sim2Real | Simulation to Real | 四足零样本迁移案例丰富 |
| WBC | Whole-Body Control | 四足亦可用全身/质心级协调 |""",
    "wiki/concepts/video-as-simulation.md": """| Vid2Sim | Video-as-Simulation | 用视频/生成模型替代或补充解析仿真 |
| WM | World Model | 从视频学环境动态的相邻概念 |
| Sim2Real | Simulation to Real | 生成资产/动力学仍须工程验收 |
| RL | Reinforcement Learning | 可在生成场景中训练策略 |
| NeRF / 3DGS | Neural Radiance Fields / 3D Gaussian Splatting | 常见场景重建与资产表示 |""",
}

BLOCK = """## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{rows}
"""


def insert_glossary(path: Path, rows: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if "英文缩写速查" in text:
        return False
    block = BLOCK.format(rows=rows.strip())

    patterns = [
        (
            r"(^## 一句话(?:定义|观点|总结)\s*\n\n.+?\n)\n(?=## )",
            r"\1\n" + block + "\n",
        ),
        (
            r"(^## 是什么\s*\n\n.+?\n)\n(?=---\s*\n\n## )",
            r"\1\n" + block + "\n",
        ),
        (
            r"(^## 核心问题（公众号分类）\s*\n\n.+?\n)\n(?=## )",
            r"\1\n" + block + "\n",
        ),
    ]
    for pat, repl in patterns:
        new_text, n = re.subn(pat, repl, text, count=1, flags=re.M | re.S)
        if n:
            path.write_text(new_text, encoding="utf-8")
            return True

    # Fallback: after first paragraph block following H1
    m = re.search(r"^(# .+\n\n)(.+?\n)\n(## )", text, re.M | re.S)
    if m and "英文缩写速查" not in m.group(2):
        new_text = text[: m.end(2)] + "\n" + block + "\n" + text[m.start(3) :]
        path.write_text(new_text, encoding="utf-8")
        return True
    raise RuntimeError(f"Could not find anchor in {path}")


def main() -> None:
    ok, skip = 0, 0
    for rel, rows in GLOSSARIES.items():
        p = ROOT / rel
        if insert_glossary(p, rows):
            print(f"INSERT {rel}")
            ok += 1
        else:
            print(f"SKIP   {rel}")
            skip += 1
    print(f"\nDone: inserted={ok} skipped={skip}")


if __name__ == "__main__":
    main()
