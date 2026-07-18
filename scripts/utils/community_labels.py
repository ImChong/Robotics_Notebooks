"""社区展示名与首页/搜索别名工具（单一事实源）。"""

from __future__ import annotations

# 社区展示名格式：「中文（English）」。规范见 schema/naming.md § 图谱社区命名。
# 社区基名默认取枢纽页 H1，但 H1 风格不一；此处按 hub 路径给出统一 override。
COMMUNITY_NAME_OVERRIDES: dict[str, str] = {
    "wiki/overview/humanoid-rl-motion-control-body-system-stack.md": (
        "人形强化学习运动控制（Humanoid Reinforcement Learning Motion Control, RL）"
    ),
    "wiki/concepts/whole-body-control.md": "全身控制（Whole-Body Control, WBC）",
    "wiki/methods/imitation-learning.md": "模仿学习（Imitation Learning, IL）",
    "wiki/tasks/locomotion.md": "运动控制（Locomotion）",
    "wiki/concepts/sim2real.md": "仿真到现实（Simulation to Reality, Sim2Real）",
    "wiki/methods/generative-world-models.md": "生成式世界模型（Generative World Models）",
    "wiki/overview/navigation-slam-autonomy-stack.md": (
        "导航与 SLAM（Navigation and Simultaneous Localization and Mapping, SLAM）"
    ),
    "wiki/entities/mujoco.md": "仿真与平台生态（Simulation and Platform Ecosystem）",
    "wiki/methods/reinforcement-learning.md": "强化学习（Reinforcement Learning, RL）",
    "wiki/queries/real-time-control-middleware-guide.md": (
        "实时运控中间件（Real-Time Control Middleware）"
    ),
    "wiki/concepts/ros2-basics.md": "机器人操作系统 2 基础（Robot Operating System 2, ROS 2）",
    "wiki/overview/motor-drive-firmware-bus-protocols.md": (
        "电机驱动器底软通信协议（Motor Drive Firmware Bus Protocols）"
    ),
    "wiki/concepts/contact-rich-manipulation.md": "接触丰富型操作（Contact-Rich Manipulation）",
    "wiki/methods/vla.md": "视觉-语言-动作（Vision-Language-Action, VLA）",
    "wiki/concepts/motion-retargeting.md": "动作重定向（Motion Retargeting）",
    "wiki/concepts/whole-body-tracking-pipeline.md": (
        "全身运动跟踪流水线（Whole-Body Tracking Pipeline, WBT）"
    ),
    "wiki/entities/humanoid-robot.md": "人形机器人（Humanoid Robot）",
    "wiki/entities/unitree-g1.md": "宇树 G1 人形机器人（Unitree G1）",
    "wiki/methods/behavior-cloning.md": "行为克隆（Behavior Cloning, BC）",
    "wiki/tasks/manipulation.md": "操作（Manipulation）",
    "wiki/tasks/teleoperation.md": "遥操作（Teleoperation）",
    "wiki/tasks/loco-manipulation.md": "移动操作（Loco-Manipulation, Loco-Manip）",
    "wiki/overview/bfm-41-papers-technology-map.md": (
        "行为基础模型技术地图（Behavior Foundation Model, BFM）"
    ),
    "wiki/overview/humanoid-motion-cerebellum-technology-map.md": "运动小脑技术地图（Motion Cerebellum）",
    "wiki/overview/humanoid-amp-motion-prior-survey.md": (
        "人形对抗式运动先验（Humanoid Adversarial Motion Prior, AMP）"
    ),
    "wiki/overview/humanoid-paper-notebooks-index.md": "人形论文深读笔记（Humanoid Paper Notebooks）",
    "wiki/overview/paper-notebook-category-01-foundational-rl.md": (
        "论文深读 · 基础强化学习（Foundational Reinforcement Learning, RL）"
    ),
    "wiki/overview/paper-notebook-category-02-motion-retargeting.md": (
        "论文深读 · 运动重定向（Motion Retargeting）"
    ),
    "wiki/overview/paper-notebook-category-03-high-impact-selection.md": (
        "论文深读 · 高影响力精选（High Impact Selection）"
    ),
    "wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md": (
        "论文深读 · 运动操作与全身控制（Loco-Manipulation and Whole-Body Control, WBC）"
    ),
    "wiki/overview/paper-notebook-category-05-locomotion.md": "论文深读 · 行走运动（Locomotion）",
    "wiki/overview/paper-notebook-category-06-manipulation.md": "论文深读 · 灵巧操作（Manipulation）",
    "wiki/overview/paper-notebook-category-07-teleoperation.md": "论文深读 · 遥操作（Teleoperation）",
    "wiki/overview/paper-notebook-category-08-navigation.md": "论文深读 · 导航（Navigation）",
    "wiki/overview/paper-notebook-category-09-state-estimation.md": (
        "论文深读 · 状态估计（State Estimation）"
    ),
    "wiki/overview/paper-notebook-category-10-sim-to-real.md": (
        "论文深读 · 仿真到现实（Simulation to Reality, Sim2Real）"
    ),
    "wiki/overview/paper-notebook-category-11-simulation-benchmark.md": (
        "论文深读 · 仿真与基准（Simulation Benchmark）"
    ),
    "wiki/overview/paper-notebook-category-12-hardware-design.md": (
        "论文深读 · 硬件设计（Hardware Design）"
    ),
    "wiki/overview/paper-notebook-category-13-physics-based-animation.md": (
        "论文深读 · 物理动画（Physics-Based Animation）"
    ),
    "wiki/overview/paper-notebook-category-14-human-motion.md": "论文深读 · 人体动作（Human Motion）",
    "wiki/concepts/foundation-policy.md": "基础策略（Foundation Policy）",
    "wiki/overview/multirotor-simulation-planning-control-stack.md": "多旋翼开源栈（Multirotor Stack）",
    "wiki/methods/sonic-motion-tracking.md": (
        "规模化运动跟踪（Supersizing Motion Tracking for Natural Humanoid Control, SONIC）"
    ),
    "wiki/overview/humanoid-hardware-101-technology-map.md": "人形硬件技术地图（Humanoid Hardware 101）",
    "wiki/overview/humanoid-actuator-102-technology-map.md": "人形执行器技术地图（Humanoid Actuator 102）",
    "wiki/overview/robot-learning-overview.md": "机器人学习（Robot Learning）",
    "wiki/methods/policy-optimization.md": "策略优化（Policy Optimization）",
    "wiki/methods/amp-reward.md": "对抗运动先验（Adversarial Motion Prior, AMP）",
    "wiki/methods/model-predictive-control.md": "模型预测控制（Model Predictive Control, MPC）",
    "wiki/entities/mimickit.md": "运动模仿与控制（MimicKit）",
    "wiki/entities/isaac-gym-isaac-lab.md": "仿真训练（Isaac Gym / Isaac Lab）",
    "wiki/tasks/humanoid-soccer.md": "人形足球（Humanoid Soccer）",
    "roadmap/depth-navigation.md": "导航纵深路线（Navigation Deep-Dive Roadmap）",
    "roadmap/depth-perceptive-locomotion.md": "感知越障纵深路线（Perceptive Locomotion Deep-Dive Roadmap）",
    "roadmap/depth-torque-motor-design.md": "力矩电机设计纵深路线（Torque-Control Motor Design Deep-Dive Roadmap）",
}

COMMUNITY_LABEL_SUFFIX = " 社区"


def community_short_label(full_label: str) -> str:
    """「中文（English） 社区」→「中文」；不合模式时返回去掉后缀的原文。"""
    base = str(full_label)
    if base.endswith(COMMUNITY_LABEL_SUFFIX):
        base = base[: -len(COMMUNITY_LABEL_SUFFIX)]
    head = base.split("（", 1)[0].strip()
    return head or base


def community_search_aliases(community_name: str) -> list[str]:
    """从社区基名「中文（English）」派生可检索别名（短中文名 + 英文副名）。"""
    base = str(community_name).strip()
    if not base:
        return []
    seen: set[str] = set()
    aliases: list[str] = []

    def add(value: str) -> None:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            aliases.append(text)

    add(community_short_label(base))
    if "（" in base and "）" in base:
        english = base.split("（", 1)[1].rsplit("）", 1)[0].strip()
        add(english)
    return aliases


def community_search_aliases_for_path(path: str) -> list[str]:
    """按 wiki 路径返回社区搜索别名（首页 chip / 图谱社区简称）。"""
    name = COMMUNITY_NAME_OVERRIDES.get(path.replace("\\", "/"))
    if not name:
        return []
    return community_search_aliases(name)
