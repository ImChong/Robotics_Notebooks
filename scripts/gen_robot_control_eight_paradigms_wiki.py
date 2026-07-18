#!/usr/bin/env python3
"""Generate wiki pages for 深蓝八大机器人控制体系 ingest."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WIKI = REPO / "wiki"
SOURCE = "../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md"
UPDATED = "2026-07-18"


def fm(
    page_type: str,
    title: str,
    tags: list[str],
    summary: str,
    related: list[str],
) -> str:
    tag_lines = "\n".join(f"  - {t}" for t in tags)
    rel_lines = "\n".join(f"  - {r}" for r in related)
    return f"""---
type: {page_type}
tags:
{tag_lines}
status: complete
updated: {UPDATED}
summary: "{summary}"
related:
{rel_lines}
sources:
  - {SOURCE}
---

"""


def tail(extra_read: str = "") -> str:
    read = extra_read or "- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>"
    return f"""
## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

{read}
"""


def method_page(
    slug: str,
    h1: str,
    one_liner: str,
    abbrev_rows: list[tuple[str, str, str]],
    tags: list[str],
    summary: str,
    related: list[str],
    why: str,
    core: str,
    practice: str,
    limits: str,
    links: str,
    extra_read: str = "",
) -> None:
    abbrev = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in abbrev_rows)
    body = f"""{fm("method", h1, tags, summary, related)}
# {h1}

{one_liner}

## 一句话定义

> {one_liner}

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

{why}

## 核心原理

{core}

## 工程实践

{practice}

## 局限与风险

{limits}

## 关联页面

{links}
{tail(extra_read)}
"""
    (WIKI / "methods" / f"{slug}.md").write_text(body, encoding="utf-8")


def overview_page(
    slug: str,
    h1: str,
    one_liner: str,
    abbrev_rows: list[tuple[str, str, str]],
    tags: list[str],
    summary: str,
    related: list[str],
    why: str,
    core: str,
    practice: str,
    limits: str,
    algo_table: str,
    links: str,
    extra_read: str = "",
) -> None:
    abbrev = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in abbrev_rows)
    body = f"""{fm("overview", h1, tags, summary, related)}
# {h1}

{one_liner}

## 一句话定义

{one_liner}

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
{abbrev}

## 为什么重要

{why}

## 核心原理

{core}

## 代表性算法

{algo_table}

## 工程实践

{practice}

## 局限与风险

{limits}

## 关联页面

{links}
{tail(extra_read)}
"""
    (WIKI / "overview" / f"{slug}.md").write_text(body, encoding="utf-8")


def main() -> None:
    tax_path = WIKI / "comparisons" / "robot-control-eight-paradigms-taxonomy.md"
    tax_path.write_text(
        fm(
            "comparison",
            "八大机器人控制体系分类",
            ["control", "taxonomy", "pid", "mpc", "rl", "robust-control", "adaptive-control"],
            "从分层闭环架构出发，将控制算法层划分为经典线性、非线性模型、鲁棒、自适应、力控、滚动优化/ILC、机器学习补偿与强化学习八类，并给出演进与融合关系。",
            [
                "../overview/robot-control-paradigm-classical-linear-feedback.md",
                "../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md",
                "../overview/robot-control-paradigm-robust-control.md",
                "../overview/robot-control-paradigm-adaptive-control.md",
                "../overview/robot-control-paradigm-hybrid-position-force.md",
                "../overview/robot-control-paradigm-receding-horizon-ilc.md",
                "../overview/robot-control-paradigm-ml-driven-control.md",
                "../overview/robot-control-paradigm-rl-intelligent-control.md",
                "../methods/pid-control.md",
                "../methods/reinforcement-learning.md",
            ],
        )
        + """# 八大机器人控制体系分类

从机器人 **任务规划 → 控制算法 → 伺服执行** 的分层闭环出发，控制算法层可划分为 **八大体系**；前四类侧重 **显式建模**，后四类分别面向 **接触作业、约束优化、数据补偿与自主习得**。

## 一句话定义

**八大控制体系** 是按建模方式、抗扰能力、数据依赖度对控制算法层的结构化分类：从 PID 等线性伺服，经 CTC/MPC 等模型方法，到阻抗/力控、ILC、神经网络补偿与 RL，形成 **经典→现代、解析→数据** 的递进关系。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PID | Proportional–Integral–Derivative | 经典线性反馈，伺服底层 |
| CTC | Computed Torque Control | 基于动力学模型的前馈+反馈 |
| MPC | Model Predictive Control | 滚动时域约束优化 |
| SMC | Sliding Mode Control | 滑模鲁棒控制 |
| RL | Reinforcement Learning | 环境交互试错学习策略 |

## 为什么重要

- **选型框架**：项目常混用 PID、MPC、RL 等名词；厘清体系边界才能判断该从哪类算法入手。
- **融合设计**：工业臂常见「CTC + SMC + ILC」、人形常见「MPC + RL」——理解单类边界是系统集成前提。
- **与具身智能衔接**：上层 VLA/模仿学习仍依赖底层伺服稳定；无 PID 电流环与中层动力学补偿，RL 难以收敛。

## 体系总览与演进

```mermaid
flowchart TB
  subgraph explicit["显式建模控制"]
    L1["① 经典线性反馈"]
    L2["② 非线性动力学控制"]
    L3["③ 鲁棒控制"]
    L4["④ 自适应控制"]
  end
  subgraph modern["现代扩展"]
    L5["⑤ 位置/力混合"]
    L6["⑥ 滚动优化与 ILC"]
    L7["⑦ 机器学习驱动"]
    L8["⑧ 强化学习智能"]
  end
  L1 --> L2 --> L3
  L2 --> L4
  L3 --> L5
  L4 --> L5
  L5 --> L6
  L6 --> L7
  L7 --> L8
```

| 序 | 体系 | 代表算法 | 独立节点 |
|----|------|----------|----------|
| 1 | [经典线性反馈](../overview/robot-control-paradigm-classical-linear-feedback.md) | [PID](../methods/pid-control.md)、[LQR](../methods/lqr-ilqr.md)、[极点配置](../methods/pole-placement-control.md) | ✓ |
| 2 | [非线性动力学控制](../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md) | [CTC](../methods/computed-torque-control.md)、[IDC](../methods/inverse-dynamics-control.md)、[反馈线性化](../methods/feedback-linearization-control.md) | ✓ |
| 3 | [鲁棒控制](../overview/robot-control-paradigm-robust-control.md) | [SMC](../methods/sliding-mode-control.md)、[H∞](../methods/h-infinity-control.md)、[μ 综合](../methods/mu-synthesis-control.md) | ✓ |
| 4 | [自适应控制](../overview/robot-control-paradigm-adaptive-control.md) | [MRAC](../methods/mrac.md)、[A-CTC](../methods/adaptive-computed-torque-control.md)、[RLS](../methods/recursive-least-squares-control.md) | ✓ |
| 5 | [位置/力混合](../overview/robot-control-paradigm-hybrid-position-force.md) | [阻抗](../concepts/impedance-control.md)、[导纳](../methods/admittance-control.md)、[力位混合](../concepts/hybrid-force-position-control.md)、[直接力反馈](../methods/direct-force-feedback-control.md) | ✓ |
| 6 | [滚动优化与 ILC](../overview/robot-control-paradigm-receding-horizon-ilc.md) | [MPC](../methods/model-predictive-control.md)、[ILC](../methods/iterative-learning-control.md) | ✓ |
| 7 | [机器学习驱动](../overview/robot-control-paradigm-ml-driven-control.md) | [NN 补偿](../methods/neural-network-compensation-control.md)、[GP](../methods/gaussian-process-control.md)、[模糊逻辑](../methods/fuzzy-logic-control.md)、[聚类故障补偿](../methods/unsupervised-clustering-fault-compensation.md) | ✓ |
| 8 | [强化学习智能](../overview/robot-control-paradigm-rl-intelligent-control.md) | [值函数 RL](../methods/value-based-reinforcement-learning.md)、[策略梯度](../methods/policy-optimization.md)、[MBRL](../methods/model-based-rl.md)、[HRL](../methods/hierarchical-reinforcement-learning.md)、[模仿学习](../methods/imitation-learning.md) | ✓ |

## 统一术语（文内）

- **系统状态**：关节角/速度/加速度、末端位姿等可量化运动量。
- **误差**：期望与实际状态之差，反馈控制核心输入。
- **扰动**：碰撞、负载变化、摩擦漂移等非预期因素。
- **前馈 / 反馈**：模型预补偿 vs 基于实时误差修正。

## 典型融合架构

- **工业机械臂**：CTC 动力学前馈 + SMC 鲁棒修正 + ILC 重复轨迹优化。
- **人形机器人**：MPC 约束步态 + RL 步态/技能优化；底层仍依赖 PD/PID 伺服。

## 常见误区

1. **「RL 可替代一切传统控制。」** 成功 RL 部署通常建立在稳定电流环与合理被控对象之上。
2. **「八类完全平行。」** 实为递进与互补；同一系统常多级联。
3. **「力控 = 阻抗。」** 阻抗、导纳、力位混合、直接力反馈适用场景不同。

## 关联页面

- [PID Control](../methods/pid-control.md)
- [Humanoid Model-based Control Stack](../overview/humanoid-model-based-control-stack.md)
- [Control Architecture Comparison](../queries/control-architecture-comparison.md)
- [WBC vs RL](../comparisons/wbc-vs-rl.md)
"""
        + tail(
            "- [depth-classical-control](../../roadmap/depth-classical-control.md)\n"
            "- [Humanoid RL Cookbook](../queries/humanoid-rl-cookbook.md)"
        ),
        encoding="utf-8",
    )

    # --- 8 paradigm overview pages ---
    overview_page(
        "robot-control-paradigm-classical-linear-feedback",
        "经典线性反馈控制（体系①）",
        "针对线性、弱扰动系统的底层伺服闭环，是电机/舵机与单关节跟踪的标配。",
        [
            ("PID", "Proportional–Integral–Derivative", "比例-积分-微分反馈"),
            ("LQR", "Linear Quadratic Regulator", "状态空间最优线性调节"),
            ("SISO", "Single Input Single Output", "单输入单输出系统"),
        ],
        ["control", "classical-control", "pid", "lqr", "linear-control"],
        "经典线性反馈是机器人伺服底层：PID/LQR/极点配置面向弱耦合线性系统，无需完整动力学模型。",
        [
            "../comparisons/robot-control-eight-paradigms-taxonomy.md",
            "../methods/pid-control.md",
            "../methods/lqr-ilqr.md",
            "../methods/pole-placement-control.md",
        ],
        "所有多关节、RL、MPC 栈最终都需 **稳定底层执行**；PID 电流/速度环是力矩输出的最后一道闸门。",
        "闭环反馈采集传感器输出与目标对比修正；LQR 在状态空间同时权衡跟踪误差与控制能耗；极点配置直接指定收敛动态。",
        "单关节标定从 P→PI→PID 逐级加项；多变量系统用 LQR 设计状态反馈增益；伺服整定用阶跃响应看超调与稳态误差。",
        "仅适用于 **弱非线性、弱耦合**；多关节强耦合需升级到非线性动力学控制。",
        "| 算法 | 节点 |\n|------|------|\n| PID | [pid-control.md](../methods/pid-control.md) |\n| LQR | [lqr-ilqr.md](../methods/lqr-ilqr.md) |\n| 极点配置 | [pole-placement-control.md](../methods/pole-placement-control.md) |",
        "- [PID Control](../methods/pid-control.md)\n- [LQR / iLQR](../methods/lqr-ilqr.md)\n- [Pole Placement Control](../methods/pole-placement-control.md)",
    )

    overview_page(
        "robot-control-paradigm-model-based-nonlinear-dynamics",
        "基于模型的非线性动力学控制（体系②）",
        "依赖精确动力学方程，用前馈抵消惯性/重力/耦合，服务多自由度机械臂与人形中层控制。",
        [
            ("CTC", "Computed Torque Control", "计算力矩/逆动力学前馈+反馈"),
            ("IDC", "Inverse Dynamics Control", "逆动力学前馈为主"),
            ("RNEA", "Recursive Newton-Euler Algorithm", "高效逆动力学递推"),
        ],
        ["control", "model-based", "dynamics", "computed-torque", "manipulation"],
        "非线性动力学控制用模型前馈将复杂系统等效线性化，CTC 是机械臂经典方案。",
        [
            "../comparisons/robot-control-eight-paradigms-taxonomy.md",
            "../methods/computed-torque-control.md",
            "../methods/inverse-dynamics-control.md",
            "../methods/feedback-linearization-control.md",
        ],
        "中高端工业臂与协作臂主流中层算法；是 WBC/TSID 与 MPC 的共同建模基础。",
        "动力学方程 $\\tau = M(q)\\ddot{q} + C(q,\\dot{q})\\dot{q} + g(q)$；CTC 用模型计算前馈力矩并叠加 PD 消除残差；反馈线性化通过微分同胚消非线性。",
        "用 [Pinocchio](../entities/pinocchio.md) 等库实时算逆动力学；标定质量/惯量参数；Sim 中验证模型失配敏感度。",
        "**模型失配**（负载变化、摩擦未建模）会显著劣化；需配合鲁棒/自适应或数据补偿。",
        "| 算法 | 节点 |\n|------|------|\n| CTC | [computed-torque-control.md](../methods/computed-torque-control.md) |\n| IDC | [inverse-dynamics-control.md](../methods/inverse-dynamics-control.md) |\n| 反馈线性化 | [feedback-linearization-control.md](../methods/feedback-linearization-control.md) |",
        "- [Computed Torque Control](../methods/computed-torque-control.md)\n- [Inverse Dynamics Control](../methods/inverse-dynamics-control.md)\n- [Feedback Linearization](../methods/feedback-linearization-control.md)",
    )

    overview_page(
        "robot-control-paradigm-robust-control",
        "鲁棒控制（体系③）",
        "在模型不精确与外部扰动下保证稳定与有界误差，强调「最坏情况」下的性能保证。",
        [
            ("SMC", "Sliding Mode Control", "滑模强制状态贴合滑模面"),
            ("H∞", "H-infinity Control", "最小化扰动到误差的最大增益"),
            ("μ", "Mu Synthesis", "结构化不确定性综合鲁棒设计"),
        ],
        ["control", "robust-control", "sliding-mode", "h-infinity", "disturbance"],
        "鲁棒控制被动抵抗扰动与参数偏差，SMC/H∞/μ 综合是工业与航空级常见工具。",
        [
            "../comparisons/robot-control-eight-paradigms-taxonomy.md",
            "../methods/sliding-mode-control.md",
            "../methods/h-infinity-control.md",
            "../methods/mu-synthesis-control.md",
        ],
        "户外移动机器人、重载臂、接触冲击场景需要 **不依赖精确模型** 的稳定性保证。",
        "SMC 设计滑模面 $s=0$ 并高频切换使状态沿面收敛；H∞ 将扰动视为输入并约束 $\\|T_{wd}\\|_\\infty$；μ 综合处理多参数不确定性块。",
        "SMC 需 **边界层/高阶滑模** 抑制抖振；H∞ 用 MATLAB Robust Control Toolbox 或 python-control 设计；与 CTC 并联作残差修正层。",
        "SMC 抖振磨损执行器；H∞/μ 设计阶次高、多关节系统标定成本大。",
        "| 算法 | 节点 |\n|------|------|\n| SMC | [sliding-mode-control.md](../methods/sliding-mode-control.md) |\n| H∞ | [h-infinity-control.md](../methods/h-infinity-control.md) |\n| μ 综合 | [mu-synthesis-control.md](../methods/mu-synthesis-control.md) |",
        "- [Sliding Mode Control](../methods/sliding-mode-control.md)\n- [H-infinity Control](../methods/h-infinity-control.md)\n- [Mu Synthesis Control](../methods/mu-synthesis-control.md)",
    )

    overview_page(
        "robot-control-paradigm-adaptive-control",
        "自适应控制（体系④）",
        "在线辨识时变参数并修正控制律，解决负载变化、磨损与摩擦漂移，与鲁棒「被动抵抗」形成互补。",
        [
            ("MRAC", "Model Reference Adaptive Control", "跟踪参考模型动态"),
            ("A-CTC", "Adaptive Computed Torque Control", "在线更新动力学参数"),
            ("RLS", "Recursive Least Squares", "递推最小二乘参数辨识"),
        ],
        ["control", "adaptive-control", "mrac", "system-identification", "parameter-estimation"],
        "自适应控制主动辨识并修正模型参数，适配变负载与工况漂移。",
        [
            "../comparisons/robot-control-eight-paradigms-taxonomy.md",
            "../methods/mrac.md",
            "../methods/adaptive-computed-torque-control.md",
            "../methods/recursive-least-squares-control.md",
        ],
        "抓取不同重量工件、长期运行摩擦变化时，固定参数 CTC 精度衰减，自适应可 **无需重标定全模型**。",
        "MRAC 用自适应律调节控制器使输出逼近参考模型；A-CTC 在线更新 $M,C,g$ 等参数；RLS 递推最小化预测误差更新参数向量。",
        "激励轨迹需 **持续激励** 保证可辨识；监控参数收敛与发散；与 RLS 辨识结果喂给前馈通道。",
        "错误激励导致参数漂移；与鲁棒层并联时需防止自适应与鲁棒项争用同一误差通道。",
        "| 算法 | 节点 |\n|------|------|\n| MRAC | [mrac.md](../methods/mrac.md) |\n| A-CTC | [adaptive-computed-torque-control.md](../methods/adaptive-computed-torque-control.md) |\n| RLS | [recursive-least-squares-control.md](../methods/recursive-least-squares-control.md) |",
        "- [MRAC](../methods/mrac.md)\n- [Adaptive CTC](../methods/adaptive-computed-torque-control.md)\n- [System Identification](../concepts/system-identification.md)",
    )

    overview_page(
        "robot-control-paradigm-hybrid-position-force",
        "位置/力混合控制（体系⑤）",
        "接触作业专用：在任务空间分解位置与力约束，实现打磨、装配、人机协作的柔顺交互。",
        [
            ("Wrench", "Force/Torque Wrench", "六维力/力矩螺旋"),
            ("F/T", "Force/Torque Sensor", "末端六维力传感"),
            ("Compliance", "Active Compliance", "主动柔顺退让"),
        ],
        ["control", "force-control", "impedance", "admittance", "contact-rich", "manipulation"],
        "力控体系解决纯位置控制在接触时的冲击问题，涵盖阻抗、导纳、力位混合与直接力反馈。",
        [
            "../comparisons/robot-control-eight-paradigms-taxonomy.md",
            "../concepts/impedance-control.md",
            "../methods/admittance-control.md",
            "../concepts/hybrid-force-position-control.md",
            "../methods/direct-force-feedback-control.md",
            "../overview/topic-contact-force-control.md",
        ],
        "插拔、恒力打磨、协作推送等任务必须在 **力维度** 上可控，否则毫米级误差即可产生破坏级接触力。",
        "阻抗调节末端等效刚度阻尼；导纳以力为输入修正位置；力位混合在约束/自由方向分别闭环位置与力；直接力反馈跟踪目标接触力。",
        "配置六维力传感器与滤波；按任务选阻抗刚度；装配任务在约束方向力控、自由方向位控。",
        "力传感噪声与延迟限制带宽；高刚度环境仍需准确环境模型或柔顺策略。",
        "| 算法 | 节点 |\n|------|------|\n| 阻抗 | [impedance-control.md](../concepts/impedance-control.md) |\n| 导纳 | [admittance-control.md](../methods/admittance-control.md) |\n| 力位混合 | [hybrid-force-position-control.md](../concepts/hybrid-force-position-control.md) |\n| 直接力反馈 | [direct-force-feedback-control.md](../methods/direct-force-feedback-control.md) |",
        "- [Impedance Control](../concepts/impedance-control.md)\n- [Topic: Contact Force Control](../overview/topic-contact-force-control.md)",
    )

    overview_page(
        "robot-control-paradigm-receding-horizon-ilc",
        "滚动优化与迭代学习控制（体系⑥）",
        "分别面向带硬约束的动态轨迹与高度重复工业轨迹，突破单步反馈的局部性。",
        [
            ("MPC", "Model Predictive Control", "滚动时域优化"),
            ("ILC", "Iterative Learning Control", "重复轨迹批次学习"),
            ("OCP", "Optimal Control Problem", "有限时域最优控制"),
        ],
        ["control", "mpc", "ilc", "optimization", "constraints", "repetitive"],
        "MPC 处理约束与多目标滚动优化；ILC 利用重复运动历史误差改进跟踪精度。",
        [
            "../comparisons/robot-control-eight-paradigms-taxonomy.md",
            "../methods/model-predictive-control.md",
            "../methods/iterative-learning-control.md",
        ],
        "人形步态、AGV 避障需要 **显式约束**；流水线搬运、定点打磨需要 **越重复越准**。",
        "MPC 每步求解有限时域 OCP 仅执行首控制量；ILC 将上批次全程误差映射为下一批次前馈补偿 $u_{k+1}=u_k + L e_k$。",
        "MPC 用 acados/forces/crocoddyl 等实时求解；ILC 需周期一致与初始条件重复；与 CTC 前馈叠加。",
        "MPC 算力与建模成本；ILC 仅适用于 **重复轨迹**，对非重复任务无效。",
        "| 算法 | 节点 |\n|------|------|\n| MPC | [model-predictive-control.md](../methods/model-predictive-control.md) |\n| ILC | [iterative-learning-control.md](../methods/iterative-learning-control.md) |",
        "- [Model Predictive Control](../methods/model-predictive-control.md)\n- [Iterative Learning Control](../methods/iterative-learning-control.md)",
    )

    overview_page(
        "robot-control-paradigm-ml-driven-control",
        "机器学习驱动控制（体系⑦）",
        "用标注数据拟合解析模型难以描述的摩擦残差、柔性形变等，常作为传统控制器的补偿模块。",
        [
            ("GP", "Gaussian Process", "非参数概率回归"),
            ("NN", "Neural Network", "万能逼近补偿网络"),
            ("LSTM", "Long Short-Term Memory", "时序依赖建模"),
        ],
        ["control", "machine-learning", "neural-network", "gaussian-process", "fuzzy-logic"],
        "机器学习控制离线/在线拟合残差动力学，为 PID/CTC 提供数据驱动补偿。",
        [
            "../comparisons/robot-control-eight-paradigms-taxonomy.md",
            "../methods/neural-network-compensation-control.md",
            "../methods/gaussian-process-control.md",
            "../methods/fuzzy-logic-control.md",
            "../methods/unsupervised-clustering-fault-compensation.md",
        ],
        "复杂摩擦、柔性关节、非标机构难以写清解析式；NN/GP 可 **贴数据补洞**。",
        "监督学习拟合 $(q,\\dot{q},\\ddot{q})\\mapsto \\tau_{res}$；GP 给出均值与不确定度；模糊逻辑编码专家规则；无监督聚类识别工况切换补偿表。",
        "采集覆盖工况的轨迹数据；残差网络与解析控制器 **并联**；GP 用于小样本安全约束控制。",
        "依赖数据分布；外推差；与 RL 不同 **不需环境试错** 但需标注或仿真标签。",
        "| 算法 | 节点 |\n|------|------|\n| NN 补偿 | [neural-network-compensation-control.md](../methods/neural-network-compensation-control.md) |\n| GP | [gaussian-process-control.md](../methods/gaussian-process-control.md) |\n| 模糊逻辑 | [fuzzy-logic-control.md](../methods/fuzzy-logic-control.md) |\n| 聚类补偿 | [unsupervised-clustering-fault-compensation.md](../methods/unsupervised-clustering-fault-compensation.md) |",
        "- [Neural Network Compensation](../methods/neural-network-compensation-control.md)\n- [Neural Feedback Controller](../concepts/neural-feedback-controller.md)",
    )

    overview_page(
        "robot-control-paradigm-rl-intelligent-control",
        "强化学习智能控制（体系⑧）",
        "无标注环境交互试错，靠奖励函数优化策略，面向未知动力学与复杂多任务。",
        [
            ("RL", "Reinforcement Learning", "强化学习"),
            ("MDP", "Markov Decision Process", "序贯决策建模"),
            ("PPO", "Proximal Policy Optimization", "工业常用策略梯度"),
        ],
        ["control", "reinforcement-learning", "policy-gradient", "model-based-rl", "hrl"],
        "RL 通过奖励驱动策略优化，涵盖值函数、策略梯度、MBRL、HRL 与模仿学习预训练。",
        [
            "../comparisons/robot-control-eight-paradigms-taxonomy.md",
            "../methods/reinforcement-learning.md",
            "../methods/value-based-reinforcement-learning.md",
            "../methods/policy-optimization.md",
            "../methods/model-based-rl.md",
            "../methods/hierarchical-reinforcement-learning.md",
            "../methods/imitation-learning.md",
        ],
        "复杂地形 loco、操作长流程任务在手工建模困难时，RL 成为 **数据驱动高层策略** 主力。",
        "MDP 上智能体选动作获奖励；值函数法估计 $Q(s,a)$；策略梯度直接优化 $\\pi_\\theta$；MBRL 先学模型再想象 rollout；HRL 分层子任务；BC/GAIL 用示教初始化。",
        "Sim 大规模训练 + DR + [Sim2Real](../concepts/sim2real.md)；实机保留 PD 执行层；PPO 为人形/足式默认首选。",
        "样本效率低、安全难、奖励设计敏感；依赖底层伺服稳定。",
        "| 分支 | 节点 |\n|------|------|\n| 值函数 | [value-based-reinforcement-learning.md](../methods/value-based-reinforcement-learning.md) |\n| 策略梯度 | [policy-optimization.md](../methods/policy-optimization.md)、[ppo.md](../methods/ppo.md) |\n| MBRL | [model-based-rl.md](../methods/model-based-rl.md) |\n| HRL | [hierarchical-reinforcement-learning.md](../methods/hierarchical-reinforcement-learning.md) |\n| 模仿学习 | [imitation-learning.md](../methods/imitation-learning.md) |",
        "- [Reinforcement Learning](../methods/reinforcement-learning.md)\n- [Humanoid RL Cookbook](../queries/humanoid-rl-cookbook.md)",
    )

    # --- algorithm method pages ---
    algorithms = [
        (
            "pole-placement-control",
            "Pole Placement Control（极点配置控制）",
            "人为指定闭环极点位置，直接塑造伺服收敛速度与振荡幅度。",
            [
                ("CL", "Closed Loop", "闭环系统"),
                ("Pole", "System Pole", "决定响应快慢与阻尼"),
                ("SISO", "Single Input Single Output", "常见标定对象"),
            ],
            ["control", "classical-control", "linear-control", "pole-placement"],
            "通过配置闭环极点位置直接塑造伺服动态响应，用于高精度伺服调校。",
            [
                "../overview/robot-control-paradigm-classical-linear-feedback.md",
                "../methods/pid-control.md",
                "../methods/lqr-ilqr.md",
            ],
            "当需要 **指定超调/调节时间** 而非仅「尽量小误差」时，极点配置比盲目调 PID 更系统。",
            "对状态空间 $(A,B)$ 若 $(A,B)$ 可控，选期望极点 $\\{p_i\\}$，求反馈 $K$ 使 $\\text{eig}(A-BK)=\\{p_i\\}$；左半平面极点保证稳定。",
            "单关节伺服用二阶近似选 $\\zeta,\\omega_n$ 映射极点；多变量用 Ackermann 公式或 place()；阶跃响应验证。",
            "依赖准确线性化模型；强非线性区需分段或增益调度。",
            "- [Classical Linear Feedback](../overview/robot-control-paradigm-classical-linear-feedback.md)\n- [PID Control](./pid-control.md)\n- [LQR / iLQR](./lqr-ilqr.md)",
        ),
        (
            "computed-torque-control",
            "Computed Torque Control（计算力矩控制，CTC）",
            "**CTC**：用动力学模型计算前馈力矩抵消非线性耦合，再叠加反馈使闭环近似线性解耦系统。",
            [
                ("CTC", "Computed Torque Control", "计算力矩控制"),
                ("FF", "Feedforward", "模型前馈"),
                ("PD", "Proportional–Derivative", "常见反馈层"),
            ],
            ["control", "model-based", "manipulation", "computed-torque", "dynamics"],
            "机械臂经典非线性控制：模型前馈线性化 + PD 反馈消除残差。",
            [
                "../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md",
                "./inverse-dynamics-control.md",
                "./feedback-linearization-control.md",
                "../entities/pinocchio.md",
            ],
            "工业协作臂标准中层方案；是理解 WBC/逆动力学控制的入门枢纽。",
            "$\\tau = M(q)(\\ddot{q}_d + K_p e + K_d \\dot{e}) + C(q,\\dot{q})\\dot{q} + g(q)$；前馈抵消 $C,g$，反馈处理模型误差。",
            "Pinocchio/RNEA 实时算 $\\tau$；Sim 标定惯性参数；与 [Friction Compensation](../concepts/friction-compensation.md) 并联。",
            "参数失配导致残余耦合；高速运动需考虑柔性未建模项。",
            "- [Model-based Nonlinear Dynamics](../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md)\n- [Inverse Dynamics Control](./inverse-dynamics-control.md)\n- [Pinocchio](../entities/pinocchio.md)",
        ),
        (
            "inverse-dynamics-control",
            "Inverse Dynamics Control（逆动力学控制，IDC）",
            "**IDC**：由期望轨迹 $(q_d,\\dot{q}_d,\\ddot{q}_d)$ 经动力学逆解直接得前馈力矩，辅以少量反馈修正。",
            [
                ("IDC", "Inverse Dynamics Control", "逆动力学控制"),
                ("RNEA", "Recursive Newton-Euler Algorithm", "O(n) 逆动力学"),
                ("CTC", "Computed Torque Control", "IDC + 闭环反馈强化"),
            ],
            ["control", "model-based", "inverse-dynamics", "manipulation"],
            "以前馈逆动力学为主、弱反馈为辅的轨迹跟踪控制。",
            [
                "../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md",
                "./computed-torque-control.md",
            ],
            "计算开销低于完整 CTC 闭环线性化时，IDC 是 **轻量前馈跟踪** 常用形态。",
            "$\\tau_{ff} = M(q)\\ddot{q}_d + C(q,\\dot{q})\\dot{q} + g(q)$；$\\tau = \\tau_{ff} + K_p e + K_d \\dot{e}$。",
            "离线轨迹规划 + 在线 IDC 前馈；反馈增益宜小，依赖模型精度。",
            "闭环修正能力弱于 CTC；突变扰动下跟踪误差更大。",
            "- [Computed Torque Control](./computed-torque-control.md)\n- [Feedback Linearization](./feedback-linearization-control.md)",
        ),
        (
            "feedback-linearization-control",
            "Feedback Linearization Control（反馈线性化控制）",
            "**反馈线性化**：通过状态反馈与坐标变换消去系统非线性，化为可控线性形式后复用 LQR/PID。",
            [
                ("FL", "Feedback Linearization", "反馈线性化"),
                ("DI", "Dynamic Inversion", "动态逆/微分同胚"),
                ("CTC", "Computed Torque Control", "机器人领域典型实现"),
            ],
            ["control", "nonlinear-control", "feedback-linearization", "model-based"],
            "通用非线性控制方法；CTC 可视为机器人动力学上的反馈线性化实例。",
            [
                "../overview/robot-control-paradigm-model-based-nonlinear-dynamics.md",
                "./computed-torque-control.md",
                "../formalizations/control-lyapunov-function.md",
            ],
            "为「先消非线性再线性控制」提供统一数学框架。",
            "寻找微分同胚 $z=\\phi(x)$ 与反馈 $u=\\alpha(x)+\\beta(x)v$ 使 $\\dot{z}=Az+Bv$；奇异点处不可线性化。",
            "验证相对阶与可控性；奇异构型附近增益调度或避障。",
            "依赖精确模型；存在内动态不稳定风险。",
            "- [Computed Torque Control](./computed-torque-control.md)\n- [Control Lyapunov Function](../formalizations/control-lyapunov-function.md)",
        ),
        (
            "sliding-mode-control",
            "Sliding Mode Control（滑模控制，SMC）",
            "**SMC**：设计滑模面 $s(x)=0$，用不连续/饱和切换律强制状态沿面滑向原点，对匹配扰动不敏感。",
            [
                ("SMC", "Sliding Mode Control", "滑模控制"),
                ("Reaching", "Reaching Law", "趋近律"),
                ("Chattering", "Control Chattering", "高频抖振"),
            ],
            ["control", "robust-control", "sliding-mode", "disturbance-rejection"],
            "工业常用鲁棒控制：滑模面约束 + 切换控制抵抗扰动与模型误差。",
            [
                "../overview/robot-control-paradigm-robust-control.md",
                "./computed-torque-control.md",
                "./h-infinity-control.md",
            ],
            "接触冲击、负载突变场景下与 CTC 并联可显著提升鲁棒性。",
            "$s = \\dot{e} + \\lambda e$；$\\tau = \\tau_{eq} - K\\,\\text{sign}(s)$；高阶/终端滑模改良收敛与抖振。",
            "边界层 $\\text{sat}(s/\\Phi)$ 减抖振；与观测器估计扰动；MATLAB/Simulink 滑模实例验证。",
            "原生 sign 切换磨损电机；需调边界层权衡鲁棒与精度。",
            "- [Robust Control Paradigm](../overview/robot-control-paradigm-robust-control.md)\n- [Computed Torque Control](./computed-torque-control.md)",
        ),
        (
            "h-infinity-control",
            "H-infinity Control（H∞ 控制）",
            "**H∞ 控制**：最小化从扰动/不确定性到跟踪误差的 **最坏情况** $H_\\infty$ 范数，保证鲁棒性能界。",
            [
                ("H∞", "H-infinity Control", "无穷范数鲁棒控制"),
                ("Riccati", "Algebraic Riccati Equation", "状态反馈求解"),
                ("LMI", "Linear Matrix Inequality", "多变量设计工具"),
            ],
            ["control", "robust-control", "h-infinity", "optimal-control"],
            "最优鲁棒控制框架，约束最坏扰动下的误差放大倍数。",
            [
                "../overview/robot-control-paradigm-robust-control.md",
                "./mu-synthesis-control.md",
                "../concepts/optimal-control.md",
            ],
            "精密手术、航空级机器人需要 **可证明** 的扰动抑制界。",
            "广义植物 $P$ 含扰动 $w$ 与性能输出 $z$；求 $K$ 使 $\\|T_{wz}\\|_\\infty < \\gamma$；Riccati/LMI 求解。",
            "线性化工作点设计 $H_\\infty$ 控制器；μ 工具分析结构不确定性；与名义控制器切换。",
            "设计保守；非线性大范围运动需增益调度或多模型切换。",
            "- [Mu Synthesis Control](./mu-synthesis-control.md)\n- [Optimal Control](../concepts/optimal-control.md)",
        ),
        (
            "mu-synthesis-control",
            "Mu Synthesis Control（μ 综合控制）",
            "**μ 综合**：在 H∞ 框架上显式处理 **结构化参数不确定性**（多关节耦合、多参数漂移），优化稳定裕度。",
            [
                ("μ", "Structured Singular Value", "结构化奇异值"),
                ("DK", "D-K Iteration", "μ 综合迭代"),
                ("H∞", "H-infinity Control", "上层鲁棒框架"),
            ],
            ["control", "robust-control", "mu-synthesis", "uncertainty"],
            "多自由度机器人多参数不确定性的进阶鲁棒设计工具。",
            ["../overview/robot-control-paradigm-robust-control.md", "./h-infinity-control.md"],
            "比单一 $H_\\infty$ 更贴合「每关节参数各不相等」的真实机构。",
            "将不确定性块 $\\Delta$ 嵌入闭环；最小化 $\\mu_\\Delta(M(j\\omega))<1$；D-K 迭代求缩放 $D$ 与控制器 $K$。",
            "MATLAB `musyn`；与多体标定数据结合界定 $\\Delta$ 范围。",
            "计算与建模成本高；工程上多用于关键子系统而非全身一次性设计。",
            "- [H-infinity Control](./h-infinity-control.md)\n- [Robust Control Paradigm](../overview/robot-control-paradigm-robust-control.md)",
        ),
        (
            "mrac",
            "MRAC（模型参考自适应控制）",
            "**MRAC**：指定理想参考模型动态，用自适应律在线调节控制器，使实际输出渐近匹配参考模型。",
            [
                ("MRAC", "Model Reference Adaptive Control", "模型参考自适应"),
                ("Lyapunov", "Lyapunov Stability", "自适应律设计依据"),
                ("MIT", "MIT Rule", "经典自适应律（历史）"),
            ],
            ["control", "adaptive-control", "mrac"],
            "经典自适应控制：参考模型对标 + 参数自适应律。",
            [
                "../overview/robot-control-paradigm-adaptive-control.md",
                "./adaptive-computed-torque-control.md",
                "../formalizations/lyapunov.md",
            ],
            "变负载抓取时无需重调固定增益，让闭环动态 **跟随理想模型**。",
            "参考模型 $\\dot{x}_m = A_m x_m + B_m r$；实际系统 $\\dot{x}=Ax+Bu$；自适应律 $\\dot{\\theta} = -\\Gamma e P b$ 使 $e=x-x_m\\to 0$。",
            "选参考模型反映期望超调/带宽；监控参数漂移；与鲁棒项并联防未建模动态。",
            "仅对满足匹配条件的扰动有效；错误参考模型导致性能劣化。",
            "- [Adaptive Control Paradigm](../overview/robot-control-paradigm-adaptive-control.md)\n- [Lyapunov](../formalizations/lyapunov.md)",
        ),
        (
            "adaptive-computed-torque-control",
            "Adaptive Computed Torque Control（自适应计算力矩，A-CTC）",
            "**A-CTC**：在 CTC 框架内在线辨识/更新惯性、质量等动力学参数，动态修正前馈补偿。",
            [
                ("A-CTC", "Adaptive Computed Torque Control", "自适应 CTC"),
                ("CTC", "Computed Torque Control", "基础框架"),
                ("RLS", "Recursive Least Squares", "常用参数更新"),
            ],
            ["control", "adaptive-control", "computed-torque", "manipulation"],
            "CTC 与自适应辨识结合，缓解变负载导致的模型失配。",
            [
                "../overview/robot-control-paradigm-adaptive-control.md",
                "./computed-torque-control.md",
                "./recursive-least-squares-control.md",
            ],
            "同一机械臂抓取不同重量工件时的 **工程常用升级路径**。",
            "用 RLS/梯度法更新 $\\hat{M},\\hat{C},\\hat{g}$；$\\tau$ 用最新参数算 CTC 律；持续激励保证收敛。",
            "抓取序列中插入辨识激励；监控 $\\hat{\\theta}$ 有界性；与固定鲁棒层并联。",
            "激励不足时参数不可辨识；与强鲁棒切换可能冲突。",
            "- [Computed Torque Control](./computed-torque-control.md)\n- [Recursive Least Squares Control](./recursive-least-squares-control.md)",
        ),
        (
            "recursive-least-squares-control",
            "Recursive Least Squares Control（RLS 递归最小二乘辨识）",
            "**RLS**：递推最小化预测误差，在线更新动力学参数估计，为自适应/前馈控制提供参数流。",
            [
                ("RLS", "Recursive Least Squares", "递归最小二乘"),
                ("SI", "System Identification", "系统辨识"),
                ("PE", "Persistent Excitation", "持续激励条件"),
            ],
            ["control", "adaptive-control", "system-identification", "rls"],
            "自适应控制的基础辨识工具，实时更新参数向量。",
            [
                "../overview/robot-control-paradigm-adaptive-control.md",
                "../concepts/system-identification.md",
                "./adaptive-computed-torque-control.md",
            ],
            "几乎所有在线自适应前馈都依赖某种 **RLS/梯度辨识** 内核。",
            "$\\hat{\\theta}_{k+1}=\\hat{\\theta}_k + K_k (y_k - \\phi_k^T \\hat{\\theta}_k)$；$K_k$ 由协方差递推得。",
            "选回归向量 $\\phi$ 含 $H(q),\\dot{q}$ 等基函数；遗忘因子应对慢时变；辨识结果喂 A-CTC。",
            "噪声与未建模动态导致偏估；需激励设计。",
            "- [System Identification](../concepts/system-identification.md)\n- [Adaptive CTC](./adaptive-computed-torque-control.md)",
        ),
        (
            "admittance-control",
            "Admittance Control（导纳控制）",
            "**导纳控制**：以测得的外力为输入，输出位置/速度修正，实现力→运动的柔顺响应，与阻抗控制对偶。",
            [
                ("Admittance", "Admittance Control", "导纳控制"),
                ("F/T", "Force/Torque Sensor", "外力测量"),
                ("Impedance", "Impedance Control", "位置→力对偶关系"),
            ],
            ["control", "force-control", "admittance", "compliance", "collaborative-robot"],
            "力输入、运动输出的柔顺控制，适合协作臂与人机交互。",
            [
                "../concepts/impedance-control.md",
                "../overview/robot-control-paradigm-hybrid-position-force.md",
                "./direct-force-feedback-control.md",
            ],
            "大负载臂、外环力控时常用 **导纳外环 + 位置内环** 结构。",
            "$M_d \\ddot{x}_c + B_d \\dot{x}_c + K_d x_c = f_{ext}$；解出修正轨迹 $x_c$ 送位置控制器。",
            "低通滤波力信号；调 $M_d,B_d,K_d$ 模拟弹簧阻尼；内环高带宽位置跟踪。",
            "依赖力传感与内环带宽；纯位置内环不佳时导纳失效。",
            "- [Impedance Control](../concepts/impedance-control.md)\n- [Hybrid Force-Position](../concepts/hybrid-force-position-control.md)",
        ),
        (
            "direct-force-feedback-control",
            "Direct Force Feedback Control（直接力反馈控制）",
            "**直接力反馈**：以目标接触力为设定值，力传感器闭环直接调节执行器输出，结构最简单。",
            [
                ("Force", "Force Control", "力闭环"),
                ("F/T", "Force/Torque Sensor", "力测量"),
                ("PI", "Proportional–Integral", "常用力环调节"),
            ],
            ["control", "force-control", "contact-rich", "manipulation"],
            "最简力控：力误差直接驱动控制量，用于恒力按压/打磨。",
            [
                "../overview/robot-control-paradigm-hybrid-position-force.md",
                "../concepts/hybrid-force-position-control.md",
                "./admittance-control.md",
            ],
            "精密按压、恒力打磨等 **单轴力跟踪** 任务足够且易调试。",
            "$e_f = f_d - f_{meas}$；$u = K_p e_f + K_i \\int e_f$；输出为速度/力矩指令。",
            "Z 轴力控打磨；滤波与饱和防冲击；与位置环分时切换。",
            "多轴耦合差；无柔顺建模时刚性碰撞仍可能过大。",
            "- [Hybrid Force-Position Control](../concepts/hybrid-force-position-control.md)\n- [Admittance Control](./admittance-control.md)",
        ),
        (
            "iterative-learning-control",
            "Iterative Learning Control（迭代学习控制，ILC）",
            "**ILC**：重复执行同一轨迹时，将上批次全程误差映射为下一批次前馈修正，迭代提升跟踪精度。",
            [
                ("ILC", "Iterative Learning Control", "迭代学习控制"),
                ("Batch", "Iteration Batch", "单次重复运行"),
                ("LTI", "Linear Time Invariant", "常见 ILC 核设计假设"),
            ],
            ["control", "ilc", "repetitive", "trajectory-tracking", "manufacturing"],
            "重复轨迹任务的批次学习优化，工业流水线常用。",
            [
                "../overview/robot-control-paradigm-receding-horizon-ilc.md",
                "./model-predictive-control.md",
                "./computed-torque-control.md",
            ],
            "与 CTC 前馈叠加可显著降低 **重复加工** 误差而不改反馈律。",
            "$u_{k+1}(t) = u_k(t) + L e_k(t)$ 或频域 $U_{k+1}=U_k + \\Gamma E_k$；$k$ 为批次索引。",
            "保证初始条件一致；选 Q-filter 保证单调收敛；与 SMC/CTC 并联。",
            "非重复轨迹无效；传感器噪声会累积到前馈。",
            "- [Receding Horizon & ILC Paradigm](../overview/robot-control-paradigm-receding-horizon-ilc.md)\n- [Model Predictive Control](./model-predictive-control.md)",
        ),
        (
            "neural-network-compensation-control",
            "Neural Network Compensation Control（神经网络补偿控制）",
            "**NN 补偿**：离线训练网络拟合动力学残差，运行时并联输出补偿力矩，修正解析模型未覆盖的非线性。",
            [
                ("NN", "Neural Network", "神经网络"),
                ("FFN", "Feedforward Network", "全连接补偿"),
                ("LSTM", "Long Short-Term Memory", "时序残差"),
            ],
            ["control", "machine-learning", "neural-network", "residual-learning"],
            "用 NN 拟合摩擦/柔性等建模残差，与传统控制器并联。",
            [
                "../overview/robot-control-paradigm-ml-driven-control.md",
                "../concepts/neural-feedback-controller.md",
                "./computed-torque-control.md",
            ],
            "复杂摩擦与非标机构上，NN 补偿是 **最低门槛** 的数据驱动增强。",
            "学习 $\\Delta\\tau = f_\\theta(q,\\dot{q},\\ddot{q})$；$\\tau = \\tau_{CTC} + f_\\theta$；可用 LSTM/Transformer 捕时序。",
            "采集多工况轨迹监督训练；注意外推安全；定期再训练。",
            "分布外工况失效；需与解析控制器并存保安全。",
            "- [ML-driven Control Paradigm](../overview/robot-control-paradigm-ml-driven-control.md)\n- [Neural Feedback Controller](../concepts/neural-feedback-controller.md)",
        ),
        (
            "gaussian-process-control",
            "Gaussian Process Control（高斯过程控制）",
            "**GP 控制**：用高斯过程建立概率动力学模型，预测下一状态并给出不确定度，支持安全约束下的决策。",
            [
                ("GP", "Gaussian Process", "高斯过程"),
                ("PILCO", "Probabilistic Inference for Learning Control", "GP+策略优化代表"),
                ("UCB", "Upper Confidence Bound", "探索-利用平衡"),
            ],
            ["control", "machine-learning", "gaussian-process", "model-based"],
            "小样本概率建模 + 不确定度量化，适合手术等低数据场景。",
            ["../overview/robot-control-paradigm-ml-driven-control.md", "./model-based-rl.md"],
            "比确定性 NN 更擅长 **小数据 + 安全界** 的控制设计。",
            "$f(x)\\sim \\mathcal{GP}(m,k)$；预测均值 $\\mu_*$ 与方差 $\\sigma_*^2$；策略优化考虑置信界。",
            "PILCO 类算法在 Sim 学 GP 动力学再优化；实机用不确定度触发保守模式。",
            "高维状态 GP 计算 $O(n^3)$；需合适核函数与诱导点近似。",
            "- [Model-based RL](./model-based-rl.md)\n- [ML-driven Control Paradigm](../overview/robot-control-paradigm-ml-driven-control.md)",
        ),
        (
            "fuzzy-logic-control",
            "Fuzzy Logic Control（模糊逻辑控制）",
            "**模糊逻辑控制**：将操作经验编码为 If-Then 模糊规则，经模糊推理与去模糊得到控制量，无需精确动力学方程。",
            [
                ("FLC", "Fuzzy Logic Control", "模糊逻辑控制"),
                ("MF", "Membership Function", "隶属度函数"),
                ("Defuzz", "Defuzzification", "模糊输出转 crisp"),
            ],
            ["control", "fuzzy-logic", "rule-based", "nonlinear-control"],
            "规则驱动的非线性控制，适合难建模非标设备。",
            ["../overview/robot-control-paradigm-ml-driven-control.md", "./pid-control.md"],
            "老师傅经验难以公式化时，模糊规则是 **可解释** 的折中。",
            "模糊化 → 规则推理 → 聚合 → 去模糊（重心法等）；可与 PID 并联调参。",
            "与专家访谈提炼规则；Sim 调隶属函数；与经典控制器切换。",
            "规则爆炸；稳定性证明难；精细任务不如模型/学习法。",
            "- [PID Control](./pid-control.md)\n- [ML-driven Control Paradigm](../overview/robot-control-paradigm-ml-driven-control.md)",
        ),
        (
            "unsupervised-clustering-fault-compensation",
            "Unsupervised Clustering Fault Compensation（无监督聚类故障补偿）",
            "**聚类故障补偿**：对运行数据无监督聚类识别工况/磨损模式，切换预存补偿参数，实现被动自适应。",
            [
                ("K-means", "K-means Clustering", "常用聚类"),
                ("Anomaly", "Anomaly Regime", "异常工况簇"),
                ("Comp", "Compensation Map", "簇对应补偿表"),
            ],
            ["control", "unsupervised-learning", "clustering", "fault-compensation"],
            "无标签工况聚类 + 补偿表切换，应对摩擦漂移与磨损。",
            [
                "../overview/robot-control-paradigm-ml-driven-control.md",
                "./neural-network-compensation-control.md",
            ],
            "长期运行机器人 **摩擦漂移** 难以手工重标定时，聚类是轻量维护手段。",
            "特征 $(\\tau,\\dot{q},e)$ 聚类 → 簇 $c$ 选补偿 $\\Delta\\tau_c$ 或增益表；在线最近簇分类。",
            "离线建簇与补偿表；在线监测簇迁移触发维护；与 ILC 批次数据结合。",
            "簇边界模糊时误切换；需足够运行数据覆盖工况。",
            "- [Neural Network Compensation](./neural-network-compensation-control.md)\n- [ML-driven Control Paradigm](../overview/robot-control-paradigm-ml-driven-control.md)",
        ),
        (
            "value-based-reinforcement-learning",
            "Value-based Reinforcement Learning（基于值函数的强化学习）",
            "**值函数 RL**：估计状态-动作价值 $Q(s,a)$ 或 $V(s)$，通过贪心或 $\\epsilon$-贪心选动作，适合离散控制。",
            [
                ("Q-Learning", "Q-Learning", "表格 Q 更新"),
                ("DQN", "Deep Q-Network", "深度 Q 网络"),
                ("TD", "Temporal Difference", "时序差分学习"),
            ],
            ["reinforcement-learning", "value-based", "dqn", "q-learning", "discrete-control"],
            "离散动作 RL：Q-Learning → DQN → Double/Dueling 变体。",
            [
                "../overview/robot-control-paradigm-rl-intelligent-control.md",
                "./reinforcement-learning.md",
                "./policy-optimization.md",
                "../formalizations/mdp.md",
            ],
            "简易 AGV、单关节离散任务仍可用；高维连续机器人主流已转向策略梯度。",
            "$Q(s,a)\\leftarrow Q + \\alpha [r + \\gamma \\max_{a'} Q(s',a') - Q]$；DQN 用 NN 近似 Q，经验回放与目标网络稳训练。",
            "离散化动作用于机械臂格点；Double DQN 减过估计；Dueling 分解 $V,A$。",
            "连续力矩控制需离散化，维数爆炸；过估计与不稳定训练。",
            "- [Reinforcement Learning](./reinforcement-learning.md)\n- [Policy Optimization](./policy-optimization.md)\n- [MDP](../formalizations/mdp.md)",
        ),
        (
            "hierarchical-reinforcement-learning",
            "Hierarchical Reinforcement Learning（分层强化学习，HRL）",
            "**HRL**：上层策略拆分子任务/选项，下层策略执行具体运动，缓解长时程信用分配与探索难题。",
            [
                ("HRL", "Hierarchical Reinforcement Learning", "分层强化学习"),
                ("Option", "Temporal Option", "时间抽象子策略"),
                ("Manager", "Manager Policy", "上层任务分配"),
            ],
            ["reinforcement-learning", "hrl", "hierarchical", "long-horizon"],
            "移动-抓取-放置等长流程任务的层次化 RL。",
            [
                "../overview/robot-control-paradigm-rl-intelligent-control.md",
                "./reinforcement-learning.md",
                "../concepts/curriculum-learning.md",
            ],
            "复杂操作任务纯扁平 RL 难收敛，HRL 提供 **时间抽象**。",
            "上层 $\\pi_{high}(o|s)$ 选 option $o$，下层 $\\pi_{low}(a|s,o)$ 执行；Option-Critic 端到端训选项。",
            "人形 loco-manip 分「行走/操作」层；分层 PPO；子目标奖励塑形。",
            "层间接口设计难；错误子目标导致下层无法完成。",
            "- [Reinforcement Learning](./reinforcement-learning.md)\n- [Curriculum Learning](../concepts/curriculum-learning.md)",
        ),
    ]

    for args in algorithms:
        method_page(*args)

    print(f"Generated taxonomy + 8 paradigms + {len(algorithms)} method pages")


if __name__ == "__main__":
    main()
