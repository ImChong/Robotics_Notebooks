# dynaretarget_arxiv_2602_06827

> 来源归档（ingest）

- **标题：** DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization
- **类型：** paper
- **来源：** arXiv abs / arXiv HTML / 项目页
- **原始链接：**
  - <https://arxiv.org/abs/2602.06827>
  - <https://arxiv.org/html/2602.06827>
  - <https://atarilab.github.io/dynaretarget.io/>
- **作者：** Victor Dhédin, Ilyass Taouil, Shafeef Omar, Dian Yu, Kun Tao, Angela Dai, Majid Khadiv（Atari Lab / TUM 等；以论文页眉为准）
- **入库日期：** 2026-06-17
- **一句话说明：** 完整 **人形 loco-manipulation 重定向管线**：IK 运动学参考 → **SBTO**（增量扩展优化时域的采样式轨迹优化）精炼为动力学可行轨迹 → **PPO 跟踪策略 + DR** 零样本上真机；在 OmniRetarget 285 条 G1–物体轨迹上成功率约 **76.8%**（SBTO_skip），约为 SPIDER SBMPC 基线 **37.9%** 的两倍。

## 核心摘录

### 1) 问题与动机
- **演示缓解 RL 探索：** 人机形态相似，可把人类动作重定向为人形参考再训 tracking policy + domain randomization 做 sim2real；但 **纯 IK 重定向**（PHC/GMR 等）易产生脚滑、穿透、缺失接触，loco-manipulation 尤甚。
- **SBMPC 局限（相对 SPIDER 等）：** 短视距 receding horizon、贪心不可回溯、轨迹抖动；长时域 loco-manipulation 上 full-horizon 一致性难保证。
- **DynaRetarget 分工：** IK 得运动学可行参考 → **SBTO** 做 dynamic refinement → RL（PPO + residual action + 物体跟踪 reward）→ 真机零样本。

### 2) SBTO（Sampling-Based Trajectory Optimization）
- **零阶优化：** 在 MuJoCo 中单次射击 rollout 评估代价 $J$，用 **CEM**（默认）更新采样分布；控制变量为 **PD 目标轨迹**，在 knot 时间 $\boldsymbol{\tau}$ 上插值。
- **增量时域：** 外环逐步增加优化 knot 数 $k=1,\ldots,K-1$；内环在当前窗口 $\tau_{k_0:\tau_k}$ 上 FHTO 直至协方差对角元 $< \sigma_{\min}$ 再 increment——**warm-start 长时域**且允许早期控制变量在更长 effective horizon（论文示例约 **3.4 s**）内继续被 refine。
- **SBTO_skip：** 对已收敛前缀缓存 rollout，跳过重复仿真，计算量约为 SBTO 的 **1/3**（Table III）。
- **代价：** 关节/基座/物体位姿与速度跟踪 + 任务空间 torso/foot/hand + 机器人–物体/自碰计数罚项（Table II）。

### 3) 实验（OmniRetarget 285 motions · G1 + box）
| 方法 | Success ↑ | Smoothness ↓ | Compute $\eta_{\text{eff}}$ ↓ |
|------|-----------|--------------|-------------------------------|
| SBTO_skip | **76.8%** | 1.41 | 1.18e7 (0.96× SPIDER) |
| SBTO | 74.6% | 1.7 | 4.06e7 |
| SBTO_pos（无速度项） | 62.1% | 2.7 | 4.46e7 |
| SPIDER (SBMPC) | 37.9% | 3.4 | 1.23e7 |

- **成功判据：** 物体平均位置误差 $<10$ cm、旋转误差 $<25°$。
- **演示增广：** 单条参考可 refine 到 **0.1–8 kg**、**0.2–0.4 m** 尺度 box，以及 cylinder / chair / shelf mesh。
- **下游 RL（Table V · 8 motions）：** DynaRetarget 参考 → 跟踪成功率 **97.09%** vs OmniRetarget  kinematic **79.41%**；训练样本效率更高（Fig. 7）。
- **真机：** 踢、搬、推、手递物体等 contact-rich loco-manipulation，零样本迁移（Fig. 1）。

### 4) 下游 RL 配方（§IV-E）
- **算法：** PPO；**mjlab**（GPU MuJoCo + IsaacLab 风格 API）；8192 envs；RTX 4090。
- **观测：** 一步期望机器人轨迹 + 物体位姿误差 + 机器人系物体位姿。
- **Reward：** DeepMimic 式 body tracking + object tracking + **contact match**（仿真内可精确提取）+ action rate / joint limit / self-collision（Table IV）。
- **DR：** 物体位姿/速度扰动、外力推、摩擦与质量随机化；**无**额外 object-tracking 早停（与 adaptive sampling 冲突）。

### 5) 代码 / 站点
- **SBTO 实现：** <https://github.com/Atarilab/sbto> — 归档见 [`sources/repos/sbto.md`](../repos/sbto.md)
- **项目页源码：** <https://github.com/Atarilab/dynaretarget.io> — 归档见 [`sources/sites/dynaretarget-github-io.md`](../sites/dynaretarget-github-io.md)
- **默认数据：** OmniRetarget HF `robot-object.zip`；Hydra 配置切换 solver / 物体 scene（box / chair / shelf / cylinder）。

## 对 wiki 的映射

- 升格 [DynaRetarget（SBTO 动力学可行重定向）](../../wiki/methods/dynaretarget-sbto-motion-retargeting.md)
- 更新 [paper-notebook-dynaretarget 实体页](../../wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md)
- 在 [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)、[SPIDER](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md)、[OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md) 中补充交叉引用

## 当前提炼状态

- [x] 摘要 + SBTO 算法 + 与 SBMPC 对比 + 下游 RL + 真机
- [x] wiki 方法页与流程图
- [x] 代码仓与项目页归档
