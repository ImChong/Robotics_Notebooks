---
type: formalization
tags: [math, probability, generative-ai, flow-matching, diffusion]
status: complete
updated: 2026-11-20
related:
  - ../concepts/flow-matching-embodied-policy.md
  - ../methods/hy-motion-1.md
  - ../methods/diffusion-motion-generation.md
  - ../overview/mit-flow-matching-diffusion-2026.md
  - ../entities/paper-reform-iclr-2026.md
sources:
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/hy_motion_arxiv_2512_23464.md
  - ../../sources/courses/mit_flow_matching_diffusion_2026.md
  - ../../sources/blogs/wechat_shenlan_flow_matching_embodied_column14_2026-09-26.md
  - ../../sources/papers/reform_iclr_2026.md
original:
  - ../../sources/courses/mit_flow_matching_diffusion_2026.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/hy_motion_arxiv_2512_23464.md
summary: "概率流（Probability Flow）提供了一个统一的数学框架来描述扩散模型与流匹配，通过将离散的去噪步转化为连续的可微常微分方程（ODE），实现了生成过程的极致加速与可控性。"
---

# Probability Flow (概率流形式化)

在具身智能的生成式动作建模（如 **π₀** 或 **Diffusion Policy**）中，**概率流 (Probability Flow)** 是连接噪声分布与真实动作分布的数学“传送带”。它将生成过程描述为一种连续的动力学系统。

## 一句话定义

通过一个与时间相关的确定性速度场 $v_\theta(x_t, t)$ 驱动常微分方程（ODE）轨迹，将简单先验噪声分布连续、可逆地传输至复杂动作数据分布的统一生成建模形式化框架。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ODE | Ordinary Differential Equation | 确定性常微分方程，用于描述连续概率传输轨迹 |
| PF-ODE | Probability Flow ODE | 与扩散 SDE 共享边际分布的确定性常微分方程 |
| FM | Flow Matching | 直接回归条件速度场或线性插值路径的无模拟训练目标 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| DiT | Diffusion Transformer | 以 Transformer 为骨干的扩散/流匹配生成架构 |
| SMPL | Skinned Multi-Person Linear Model | 常见人体参数化模型与重定向源 |

## 为什么重要

1. **推理速度与确定性求解**：传统 DDPM/SDE 常需要数十到数百步随机迭代；概率流 ODE 与直线化流匹配轨迹配合 Heun、Euler 或 RK45 等数值求解器，可在极少积分步内完成高质量动作采样，满足机器人控制高频需求。
2. **时域平滑与几何可控**：生成轨迹由连续速度场积分得到，天然兼顾动作段内的平滑过渡；在直接动作空间（如 [MotionSpaceFlow](../entities/paper-motionspaceflow.md)）中还能利用干净终点估计 $\hat{x}_1$ 直接做零样本端点或路点投影控制。
3. **统一 Diffusion 与 Flow**：得分匹配（Score Matching）与最优传输（OT）线性条件流匹配在边缘密度演化上互为对偶，便于在策略与动作模型之间迁移采样调度与偏好优化方法（如 Flow-GRPO）。

## 核心原理

给定从简单噪声分布 $p_0$（通常为各向同性高斯分布）到复杂动作分布 $p_1$ 的变换，概率流通过如下常微分方程（ODE）描述样本 $x_t$ 随时间 $t \in [0, 1]$ 的演化：

$$ dx_t = v(x_t, t)\, dt $$

其中 $v(x, t)$ 是 **速度场 (Velocity Field)**。

### 1. 与扩散模型 (Diffusion SDE) 的等价边际演化

在扩散模型中，Fokker–Planck 方程保证了概率流 ODE (PF-ODE) 能够以确定性方式复现对应随机微分方程 (SDE) 的每一时刻边际分布 $p_t(x)$。已知得分函数 $\nabla_x \log p_t(x_t)$ 与漂移/扩散系数 $f, g$ 时，速度场可写为：

$$ v(x_t, t) = f(x_t, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x_t) $$

### 2. 条件流匹配 (Conditional Flow Matching, CFM)

以最优传输（OT）线性插值路径 $x_t = t x_1 + (1 - t) x_0$ 为例，模型不再经过长马尔可夫去噪链，而是直接回归条件速度场目标 $x_1 - x_0$：

$$ \mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right] $$

当网络改为直接预测干净终点 $\hat{x}_1$（常见于高维人体动作生成如 HY-Motion 与 MSFlow）时，瞬时速度由 $(\hat{x}_1 - x_t)/(1 - t)$ 换算获得。

### 3. 动作生成中的尺度与偏好对齐

- **十亿级 T2M 验证**：[HY-Motion 1.0](../methods/hy-motion-1.md)（arXiv:2512.23464）把 **DiT + Flow Matching** 推至十亿参数级，借助 GVHMR → SMPL-H（`>3000` 小时语料）及 OT 常数速度场 + ODE 积分，并用 Flow-GRPO 将强化学习偏好引入流轨迹末端对齐。
- **直接空间与支撑集构造**：下表概括了动作域常见流公式变体：

| 路线 | 关键机制 | 代表 |
|------|----------|------|
| 终点 $\hat{x}_1$ 参数化 + 尺度 $s$ | 对高斯源加尺度并裁剪 $1-t$ 下限 | [MotionSpaceFlow](../entities/paper-motionspaceflow.md)（直出 263D / XYZ66D） |
| 距离 / 保守 Q 正则 | 惩罚偏离数据集或 OOD Q 估计 | CQL、IQL、TD3+BC |
| **Support-by-construction** | 有界源 BC flow 定义 support，再在 support 内操纵噪声 | [ReFORM](../entities/paper-reform-iclr-2026.md)（ICLR 2026，OGBench） |

## 工程实践

- **$x_1$-prediction 与终点数值稳定**：当直接预测干净动作 $\hat{x}_1$ 再反推速度 $(\hat{x}_1 - x_t)/(1-t)$ 时，须在 $t \to 1$ 附近对分母 $1-t$ 做下界裁剪，并在采样最后一步采用 Euler 替代高阶校正以免高曲率放大误差。
- **步数预算与求解器**：机器人闭环场景常选 1–10 步 Euler/Heun 配合蒸馏或一致性策略；离线高保真人体生成（如 HY-Motion、MSFlow）多采用 20–50 步 Heun/ODE 积分及 Classifier-Free Guidance（CFG）。
- **物理后验衔接**：确定性 ODE 保证采样均值平滑，但不编码接触动力学或关节扭矩限制；用于真机时仍需下游跟踪控制器（如 [SONIC](../entities/nvidia-sonic.md)、[Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md) 中的闭环方案）或 RL tracker 滤波修正。

## 局限与风险

- **支撑集外推与均值坍缩**：线性插值轨迹在少步推理下容易在多模态歧义动作（如左转 vs 右转）间发生平均化（mode averaging），导致生成姿态穿透或失去锐度。
- **分布支撑外 OOD 失控**：标准高斯先验无界，一旦约束或引导信号将 $x_t$ 拉出训练轨迹支撑集，数值积分会沿未训练速度场发散。
- **$C^1$ 运动学连续 $\neq$ 物理可行**：ODE 输出平滑的关节坐标或旋转曲线，并不意味着地面反作用力、摩擦锥或电机扭矩在实机安全包络内。

## 关联页面
- [流匹配与具身策略（概念）](../concepts/flow-matching-embodied-policy.md) — VLA 动作头 FM 的工程口径（专栏编译）
- [MIT 6.S184 Flow Matching & Diffusion（2026）](../overview/mit-flow-matching-diffusion-2026.md) — SDE/FM 系统课
- [HY-Motion 1.0](../methods/hy-motion-1.md) — 十亿级 DiT + 流匹配文本到 SMPL-H 生成
- [MotionSpaceFlow](../entities/paper-motionspaceflow.md) — 直接动作空间流匹配与 $s$-scaled 噪声分析
- [π₀ (Pi-zero) 策略模型](../methods/pi0-policy.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源
- [mit_flow_matching_diffusion_2026.md](../../sources/courses/mit_flow_matching_diffusion_2026.md) — MIT 6.S184 SDE/FM 系统课
- [diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — 扩散与生成模型基础讲义/论文摘录
- Lipman, Y., et al. (2022). *Flow Matching for Scalable Simulation*.
- Song, Y., et al. (2020). *Score-Based Generative Modeling through Stochastic Differential Equations*.
- [sources/papers/hy_motion_arxiv_2512_23464.md](../../sources/papers/hy_motion_arxiv_2512_23464.md) — HY-Motion 1.0
- [reform_iclr_2026.md](../../sources/papers/reform_iclr_2026.md) — ReFORM：reflected flow + offline RL support

## 推荐继续阅读

- [MIT 6.S184 Generative AI with Stochastic Differential Equations](https://diffusion.csail.mit.edu/2026/) — 官方课程讲义
- [MotionSpaceFlow（arXiv:2609.34190）](https://arxiv.org/abs/2609.34190) — 动作表示与噪声尺度对概率流条件数的实证分析
