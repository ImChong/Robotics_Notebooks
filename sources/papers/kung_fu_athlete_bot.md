# KungFuAthleteBot: Highly Dynamic Martial Arts Motion Dataset and Autonomous Fall-Resilient Tracking

> 来源归档（ingest）

- **标题：** A Kung Fu Athlete Bot That Can Do It All Day: Highly Dynamic, Balance-Challenging Motion Dataset and Autonomous Fall-Resilient Tracking
- **类型：** paper / dataset / repo
- **arXiv abs：** <https://arxiv.org/abs/2602.13656>
- **arXiv HTML：** <https://arxiv.org/html/2602.13656v1>
- **PDF：** <https://arxiv.org/pdf/2602.13656>
- **项目页：** <https://kungfuathletebot.github.io/>
- **代码仓库：** <https://github.com/NPCLEI/KungFuAthleteBot>
- **作者：** Zhongxiang Lei, Lulu Cao, Xuyang Wang, Tianyi Qian（✉）, Jinyan Liu（✉）, Xuesong Li（✉）
- **机构：** 北京理工大学（BIT）、启元实验室（QIYUAN Lab）
- **硬件：** Unitree G1（29 DoF，约 1.3 m）
- **仿真栈：** Isaac Sim 5.0 训练；MuJoCo 评测；FastSAC off-policy RL；单卡 NVIDIA A100 80GB
- **入库日期：** 2026-05-04（初稿）；**最后更新：** 2026-07-02（arXiv / 项目页 / 仓库深读升格）
- **一句话说明：** 从国家级武术运动员日常训练视频构建 **KungFuAthlete** 高动态数据集（848 样本 / Ground+Jump 子集），提出 GVHMR→GMR 后 **根高度漂移抛物线校正 + SG 平滑** 管线，并用 **单策略 FastSAC** 联合高动态 motion tracking 与 **GRSI 重力随机跌倒恢复**，在 G1 真机实现抗扰长时程武术跟踪。

## 摘要级要点

- **问题缺口：** 现有 tracking（GMT、TWIST、BeyondMimic 等）在 **硬件性能极限与算法鲁棒边界** 附近仍不足；武术 = 极速质心转移 + 复杂协调 +  abrupt 姿态切换，且 **专业运动员也会失误**——多数工作假设全程安全态，缺统一 **unsafe 态建模 + 自主恢复**。
- **数据集 KungFuAthlete：** 197 原视频 → 自动切分 1,726 子片段 → GVHMR + GMR → 过滤后 **848** 样本；**Ground**（非跳跃，约 84% 日常训练）与 **Jump**（空翻、旋子等）子集；Jump 子集关节/线/角速度显著高于 LAFAN1、PHUMA、AMASS。
- **数据校正：** 针对视频重建 **根节点高度漂移**（悬浮/穿地）与帧间 **局部抖动**——分段地面接触约束 + 速度阈值传播 + 跳跃段 **抛物线重建** + 最终穿地修正；Savitzky–Golay 平滑保峰值。
- **训练范式：** 单策略最大化 $r_{\mathrm{mt}} + \mathbb{I}_{\mathrm{rc}} r_{\mathrm{rc}}$；Bernoulli 混合 motion tracking 与 recovery episode；**低动能采样（LKE）** 锚点初始化避免空中不可恢复态；**GRSI** 零力矩重力释放 + 姿态重组扩覆盖跌倒初态。
- **项目状态（2026-02）：** Ground 子集 largely ready；Jump 子集因视频源仍有 minor imperfections；模型训练 active development。

## 核心摘录（面向 wiki 编译）

### 1) 数据集规模与动力学对比（§3.1, Table 1）

| 数据集 | FPS | Joint Vel. | Body Lin. Vel. | Body Ang. Vel. | Avg Frames |
|--------|-----|------------|----------------|----------------|------------|
| LAFAN1 | 50 | 0.00142 | 0.00021 | 0.01147 | 10749 |
| PHUMA | 50 | 0.00120 | 0.00440 | -0.00131 | 170 |
| AMASS | 30 | 0.00048 | -0.00568 | 0.00903 | 371 |
| **KungFuAthlete (Ground)** | 50 | -0.00199 | 0.01057 | 0.04034 | 578 |
| **KungFuAthlete (Jump)** | 50 | **0.02384** | **0.05297** | **0.18017** | 397 |

- **类别分布：** Daily Training 715；Fist 53（长拳 33）；Staff 30；Skills 28（后空翻 12、旋子 9）；Saber 15；Tai Chi Sword 7。
- **对 wiki 的映射：**
  - [KungFuAthlete（论文实体）](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)
  - [humanoid-reference-motion-datasets](../../wiki/comparisons/humanoid-reference-motion-datasets.md)

### 2) 根高度漂移校正（§3.2, Algorithm 1）

- **链接：** arXiv §3.2–3.3；项目页 Dataset Overview
- **摘录要点：**
  - 最小体高约束 $z_{\min}(\mathbf{q}_t)$ 对齐支撑相地面；非接触相用速度阈值 $\tau$ 传播 $\hat{P}(t)$ 抑制抖动。
  - 跳跃段：局部极值检测 take-off/landing，中间 **抛物线** $y(t)=y_0-\frac{1}{2}gt^2$ 重建 airborne 根轨迹。
  - SG 滤波（窗口约序列长度 1/10，奇数）平滑关节角与根位，保留 apex/接触峰值。
- **对 wiki 的映射：**
  - [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) — 视频→人形参考的数据清洗层
  - [KungFuAthlete](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md) — 管线 Mermaid

### 3) 单策略 tracking + 自主恢复（§4, Eq. 14）

- **链接：** arXiv §4.1–4.5
- **摘录要点：**
  - 目标：$\max_\pi \mathbb{E}[\sum \gamma^k (r_{\mathrm{mt}} + \mathbb{I}_{\mathrm{rc}} r_{\mathrm{rc}})]$；$p$ 概率做 tracking，$1-p$ 做 recovery；recovery 指示 $\mathbb{I}_{\mathrm{rc}}$ 由肩高偏差 $>\tau$ 触发。
  - **LKE 采样：** 参考轨迹动能代理 $E(t)=\sum_j|\dot{q}^{\mathrm{ref}}_{t,j}|$ 的局部极小为 episode 锚点；失败归因最近前锚并增权（对比 BeyondMimic 失败驱动采样在腾空相的不可恢复问题）。
  - **GRSI：** 零力矩 + 重力释放 + 随机摩擦 → 多样跌倒姿态 $\mathcal{D}_{\mathrm{GRSI}}$；旋转分量重组增覆盖。
  - **FastSAC** off-policy 并行仿真训练；域随机化沿用 FastSAC 协议（质量、摩擦、CoM、PD、延迟、在线 push）。
- **对 wiki 的映射：**
  - [Balance Recovery](../../wiki/tasks/balance-recovery.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [SafeFall](../../wiki/entities/paper-hrl-stack-41-safefall.md)、[BeyondMimic](../../wiki/entities/paper-notebook-kungfubot-2.md) 对照语境

### 4) 奖励与终止（§4.3–4.4）

- **$r_{\mathrm{mt}}$（Table 2）：** BeyondMimic 式相对体位/朝向/角速度 + HuB 式 CoM–支撑足对齐 + 强 **feet slip**（$z$ 接触力 >8 时罚 $xy$ 脚速）+ 膝/踝 action rate + 非期望接触。
- **$r_{\mathrm{rc}}$（Table 3）：** 站起前罚肩高偏差、$xy$ 根位移、动作突变——鼓励 **原地** 平滑恢复。
- **终止：** recovery 模式下允许连续 bad tracking 步数 $L(k)\geq\tau_{\mathrm{bad}}$ 才终止，避免恢复中途过早截断。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 5) 仿真消融与真机（§5）

- **平台：** Isaac Sim 5.0 训练，MuJoCo 评测；G1 29 DoF 真机部署。
- **指标：** Succ.（关节位误差 >0.5 m / 朝向 >0.8 rad / 摔倒即失败）、$E_{\mathrm{mpboe}}$、Smooth（action rate）。
- **消融（1307 帧高难度序列，6 trials）：**

| Method | $E_{\mathrm{mpboe}}$ | Succ. | Smooth |
|--------|---------------------|-------|--------|
| BeyondMimic | 12.55 | **0/6** | 14.2 |
| w/ recover task | 40.39 | **6/6** | 18.6 |
| + feet slip (w=3) | 25.27 | 6/6 | 19.2 |
| + feet slip (w=5) | 19.93 | 6/6 | 18.0 |

- **定性：** 加 recovery 目标后 BeyondMimic 基线可在无扰单腿站 **不再全败**；feet slip + CoM + 膝踝 rate 改善交叉步抬脚质量。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)

### 6) 与 KungfuBot / KungfuBot2 谱系

- **KungfuBot**（arXiv:2506.12851）：同类武术高动态 tracking，物理 COM 滤波 + curriculum + domain randomization。
- **KungfuBot 2**（arXiv:2509.16638）：教师–学生蒸馏 + ACT/BC，Versatile motion skills。
- **本文差异：** 强调 **数据集动力学上界**（运动员日常视频 + Jump 子集统计）+ **单策略 tracking∪recovery**（无需参考起身动作）+ **GVHMR 漂移校正** 工程管线。
- **对 wiki 的映射：**
  - [paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont](../../wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md)
  - [paper-notebook-kungfubot-2](../../wiki/entities/paper-notebook-kungfubot-2.md)

## 对 wiki 的映射（汇总）

- 升格实体页：[paper-kungfuathlete-humanoid-martial-arts-tracking.md](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)
- 仓库归档：[kungfuathletebot.md](../repos/kungfuathletebot.md)
- 交叉：[balance-recovery](../../wiki/tasks/balance-recovery.md)、[motion-retargeting](../../wiki/concepts/motion-retargeting.md)、[humanoid-reference-motion-datasets](../../wiki/comparisons/humanoid-reference-motion-datasets.md)

## 参考来源（原始）

- arXiv:2602.13656 — 论文正文
- 项目页：<https://kungfuathletebot.github.io/>
- 代码：<https://github.com/NPCLEI/KungFuAthleteBot>
- 视频素材授权：谢远航（广西武术队，国家级运动健将，中国武术六段）
