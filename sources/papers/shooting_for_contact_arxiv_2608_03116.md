# Shooting for Contact: Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting（arXiv:2608.03116）

> 来源归档（ingest · 全文消化）

- **标题：** Shooting for Contact: Contact-Implicit Multiple Shooting for Dynamic Motion Retargeting
- **类型：** paper / contact-implicit trajectory optimization + motion retargeting + motion-imitation RL
- **arXiv abs：** <https://arxiv.org/abs/2608.03116>
- **arXiv HTML：** <https://arxiv.org/html/2608.03116v1>
- **PDF：** <https://arxiv.org/pdf/2608.03116>；项目页镜像 <https://shooting-for-contact.github.io/static/pdfs/Contact_Rich_Locomotion.pdf>
- **项目页：** <https://shooting-for-contact.github.io/>（归档见 [`sources/sites/shooting-for-contact-github-io.md`](../sites/shooting-for-contact-github-io.md)）
- **代码：** <https://github.com/sesteban951/shooting-for-contact>（归档见 [`sources/repos/shooting-for-contact.md`](../repos/shooting-for-contact.md)）— **已开源**（MuJoCo + IPOPT DSMS 参考实现与 G1/Go2 示例）
- **作者：** Sergio A. Esteban, Jason H. K. Siu, Derrick Mach, Junheng Li, Vince Kurtz, Joel W. Burdick, Aaron D. Ames
- **机构：** 加州理工学院（Caltech）；德保罗大学（DePaul University，Vince Kurtz）
- **硬件：** Unitree G1（主真机）；Unitree Go2（形态无关示例）
- **入库日期：** 2026-08-07
- **一句话说明：** 用可微仿真器离散转移映射作动力学的 **接触隐式直接仿真多重打靶（DSMS）** NLP，把运动学可行参考转为 **全身动力学可行** 轨迹（无需接触时刻表 / 显式接触约束），加速下游 motion-imitation RL，并在 G1 上零样本部署 **命令条件化爬行** 与 **180° 跳转**。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://shooting-for-contact.github.io/> | 方法示意、真机爬行/跳转、动态接触动作库视频 |
| 代码 | <https://github.com/sesteban951/shooting-for-contact> | DSMS NLP（`multi_shooting.py`）+ MPC + G1/Go2 examples |
| 运动学对照 | [OmniRetarget](omniretarget_arxiv_2509_26633.md) | Sequential SOCP；动力学可行性弱 |
| 采样动力学对照 | [DynaRetarget / SBTO](dynaretarget_arxiv_2602_06827.md) | CEM + MuJoCo rollout；任意硬约束弱 |
| 下游 RL 配方先例 | BeyondMimic 系 | 论文 motion-imitation / crawling controller 对齐 mjlab tracking |

## 摘要级要点

- **痛点：** 常见重定向优先运动学相似度，忽略全身动力学、接触一致性与作动极限 → RL 难复现，接触丰富行为尤甚。
- **方法：** **DSMS** — 多重打靶 NLP，区间内用可微仿真器（MuJoCo）rollout；接触/摩擦/冲击/自碰/关节限在仿真器内解析；NLP 只强制跟踪、作动与任务约束。
- **管线：** 运动学/降阶参考 → DSMS（全时域或 receding-horizon）→ 动力学可行参考 → mjlab PPO motion imitation / 命令条件化 gait library 跟踪。
- **真机：** G1 **零样本** — twist 命令爬行（室内限高、室外草地坡）；**180° jump-turn**（SRB→DSMS）。
- **对比：** Super-hero backflip 上相对 OmniRetarget / BONES-SEED / DynaRetarget，DSMS 训练收敛更快；落地成功率与 DynaRetarget 同档（74/75），显著高于 OmniRetarget（7/75）。

## 核心摘录（面向 wiki 编译）

### 1) DSMS NLP（§III-A）

- 时域划为 \(N\) 个 shooting intervals；决策变量为 shooting-node 状态 \(\mathbf{X}\) 与控制 spline 点 \(\mathbf{U}\)。
- 缺陷约束 \(\mathbf{x}_{k+1}=\mathbf{F}(\mathbf{x}_k,\mathbf{u}_k)\)：\(\mathbf{F}\) 为仿真器离散转移，内含 \(N_s\) 细步；接触刚度在细步内解析，NLP 保持小规模。
- **接触隐式：** 无接触 schedule、无互补松弛、无接触力决策变量。
- 求解：`cyipopt` / IPOPT + HSL `ma57`；MuJoCo 一阶差分梯度 + L-BFGS 曲率近似。

### 2) 代价与约束（§III-B–D）

- 代价：状态跟踪 + key-body 位姿/twist 跟踪 + 实现力矩正则 + 命令变化率正则。
- 高动态动作（后空翻、翻滚）：**receding-horizon** 重复求解拼接，保证整段由仿真器生成而动力学可行。
- **Gait synthesis：** 单周期 NLP + limit-cycle 闭合（平面位姿按命令推进，其余状态周期返回）；软 no-slip 罚允许爬行滑动接触；扫描 twist 网格得命令–步态库。

### 3) 下游 RL（§IV）

- **Motion imitation：** BeyondMimic 风格 body pose/twist reward、RSI、adaptive frame sampling；mjlab + rsl_rl PPO。
- **Locomotion controller：** 非对称 actor–critic；actor 仅本体感知 + twist 命令 + gait phase；critic 额外见参考；离散库充当连续速度接口。
- **Sim2Real DR：** base push、COM 偏移、关节编码器偏置、接触摩擦。

### 4) 实验数字（§V · HTML 摘录）

**Table I · ROM 动态保真消融（sim-to-sim 落地）：**

| 方法 | Landed Success |
|------|----------------|
| SRB | 0/75（0%） |
| SRB → KD | 0/75（0%） |
| SRB → **DSMS** | **75/75（100%）** |
| SRB → KD → DSMS | 72/75（96.0%） |

**Table II · Super-hero backflip 重定向对比：**

| 方法 | Landed Success |
|------|----------------|
| OmniRetarget (OR) | 7/75（9.3%） |
| BONES-SEED (BS) | 60/75（80.0%） |
| DynaRetarget (DR) | 74/75（98.7%） |
| **DSMS** | **74/75（98.7%）** |

- DSMS 训练曲线最快收敛；相对最近竞争者约 **40 min** 墙钟优势（RTX 4090）。
- **Table III 定性：** DSMS 同时具备 whole-body 动力学可行、接触隐式、任意等式/不等式约束；相对 OmniRetarget（缺 WB dyn.）、DynaRetarget（缺 arbitrary const.）、SPARK（非接触隐式）。

### 5) 开源状态（项目页核查 · 2026-08-07）

- 项目页 **Code** 链到 <https://github.com/sesteban951/shooting-for-contact>。
- 仓库含可运行 `examples/`（cartpole、hopper、Go2 tracking MPC、G1 gait / squat / tracking MPC）与 `src/multi_shooting.py`；**RL 训练栈与真机部署脚本未随仓发布**（论文用 mjlab；仓聚焦 DSMS trajopt/MPC）。
- License：截至入库日 GitHub API **未列出** SPDX license 文件。

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-shooting-for-contact.md`](../../wiki/entities/paper-shooting-for-contact.md)
- 方法页：[`wiki/methods/dsms-contact-implicit-multiple-shooting.md`](../../wiki/methods/dsms-contact-implicit-multiple-shooting.md)
- 交叉：[`wiki/concepts/motion-retargeting.md`](../../wiki/concepts/motion-retargeting.md)、[`wiki/overview/hub-motion-retargeting.md`](../../wiki/overview/hub-motion-retargeting.md)、[`wiki/methods/dynaretarget-sbto-motion-retargeting.md`](../../wiki/methods/dynaretarget-sbto-motion-retargeting.md)、[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)

## 关联原始资料

- 项目页：[`sources/sites/shooting-for-contact-github-io.md`](../sites/shooting-for-contact-github-io.md)
- 代码：[`sources/repos/shooting-for-contact.md`](../repos/shooting-for-contact.md)
