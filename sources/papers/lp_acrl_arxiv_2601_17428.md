# Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning（arXiv:2601.17428）

> 来源归档（ingest · 一手论文）

- **标题：** Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning
- **短名：** LP-ACRL
- **类型：** paper / quadruped / curriculum-rl / automatic-curriculum / anymal / isaac-lab
- **arXiv：** <https://arxiv.org/abs/2601.17428>
- **arXiv HTML：** <https://arxiv.org/html/2601.17428>
- **PDF：** <https://arxiv.org/pdf/2601.17428>
- **DOI：** <https://doi.org/10.1109/LRA.2026.3703486>
- **项目页：** <https://sites.google.com/view/lp-acrl>
- **RSL 出版物索引：** <https://rsl.ethz.ch/publications-sources/publications.html>
- **作者：** Ziming Li、Chenhao Li、Marco Hutter
- **机构：** Robotic Systems Lab, ETH Zurich；Chenhao Li @ ETH AI Center
- **发表：** IEEE Robotics and Automation Letters（RA-L），2026
- **入库日期：** 2026-10-01
- **一句话说明：** 用 episodic reward 的 **learning progress（LP）** 在线 softmax 重采样离散任务实例，在 **600 实例** 多轴 velocity×terrain×难度空间自动课程 PPO；Teacher 高程图策略经 **Teacher–Student 蒸馏**（LSTM+MLP 学生）部署 ANYmal D，rough terrain **2.5 m/s** 线速度、**3.0 rad/s** 角速度。

## 开源状态（步骤 2.5）

- **核查日：** 2026-10-01；见 [`sources/sites/lp-acrl.md`](../sites/lp-acrl.md)。
- **邻近开源（非本文实现）：** [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) — RSL on-GPU PPO 库；[leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) — 经典地形课程并行训练（**手工** 难度轴，非 LP-ACRL）。
- **结论：** **确认未开源** LP-ACRL 课程模块与权重。

## 摘录 1：问题与 LP-ACRL 机制（§III）

**动机：** 真实 legged 任务常为 **多 categorical + 多 continuous 轴** 的笛卡尔积（地形类型 × 几何难度 × 线/角速度档位），**不存在** 单一可手工排序的难度轴；手工 CRL（Rudin 地形晋级、Ji 速度上界递增、Margolis grid expansion、Li et al. 十级压缩轴）在 **600+ 实例** 空间不可维护。

**离散任务空间：** 连续维（如 $|\mathbf{v}_{b,x}^*|$）划分子区间； categorical 维（楼梯上/下、 gravel 等）直接枚举；任务实例 $\zeta\in\mathcal{T}$ 为组合。

**LP 与采样更新：**

- 期望 episodic reward：$R_{c_j}(\zeta)=\mathbb{E}_{\tau\sim c_j}[R_\tau]$
- Learning progress：$LP_{c_j}(\zeta)=R_{c_j}(\zeta)-R_{c_{j-1}}(\zeta)$
- 下一阶段分布：$c_{j+1}(\zeta)\propto \exp(LP_{c_j}(\zeta)/\beta)$（softmax 温度 $\beta$）

**解读：** 正 LP → 仍在上坡的任务获更高采样； plateau 后概率 **回流** 到仍有 LP 的中低难任务，减轻 **catastrophic forgetting**；相对 **ALP**（|LP| 放大回归波动）与 **PLR**（value error 任务区分弱）更适配多轴 locomotion。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-lp-acrl-scaling-rough-terrain-locomotion.md`](../../wiki/entities/paper-lp-acrl-scaling-rough-terrain-locomotion.md)；概念 [`wiki/concepts/curriculum-learning.md`](../../wiki/concepts/curriculum-learning.md)「自动课程」节。

## 摘录 2：三层实验与 EPTE-SP（§IV）

**基线：** ALP [Portelas et al.]、PLR [Jiang et al.]、Simple Hand-Crafted (SC) [Ji et al.]、LRPC、Uniform。

**指标 EPTE-SP $\tilde{\varepsilon}$：** 联合速度跟踪百分比误差与 **早停惩罚**（跌倒后剩余步计最坏误差），越低越好。

| 实验 | 任务空间规模 | LP-ACRL 要点 |
|------|-------------|--------------|
| IV-C 平地线速度 | 8 档 $\lvert v_x^*\rvert\in[0,4]$ m/s | EPTE-SP 与 reward 收敛均最优；采样热图显示 **先易后难再回流** |
| IV-D 多地形 | 6 地形 × 随机 $v_x,v_y,\omega_z\in[0,1]$ | 无显式难度序仍优于基线 |
| IV-E scaled | **600** 实例（5×6 速度档 × 5 地形 × 4 几何难度） | **1500 iter 达 ~80% success**；基线 3000 iter 仍难收敛 |

**Success 定义（IV-E）：** episode >900 alive steps 且 $v_x^*,\omega_z^*$ 的 EPTE-SP 均 <30%。

**对 wiki 的映射：** 与 [Parkour in the Wild](../../wiki/entities/paper-parkour-in-the-wild.md) 等同 ANYmal 栈的 **扩展任务空间** 对照；仿真侧 [Isaac Lab](../../wiki/entities/isaac-lab.md)。

## 摘录 3：真机 Teacher–Student 与速度 claims（§IV-E 末）

- **Teacher：** 108/275 维局部 **height map** 网格（2.4 m × 1.0 m，0.1 m 分辨率）+ 本体 + 速度指令；LP-ACRL 训练。
- **Sim2Real 噪声：** 真机 height map 由 elevation mapping 重建，高速时噪声大。
- **Student：** **LSTM + MLP** 利用时序，从 teacher **蒸馏** 后部署 ANYmal D。
- **报告速度：** 平地 **3.0 m/s**；楼梯/坡/ gravel 等 **2.5 m/s**；各 terrain **3.0 rad/s** 角速度跟踪。

**对 wiki 的映射：** [Privileged Training](../../wiki/concepts/privileged-training.md) teacher-student 读法；邻近 [RSL RL](../../wiki/entities/rsl-rl.md) 训练栈。

## 摘录 4：与手工地形课程 lineage（Related §II-A）

训练框架 **follow Rudin et al. [4] and Schwarke et al. [21]**（parallel env、rough terrain reward/obs 惯例）；LP-ACRL **替换** 的是 **任务实例采样 $c_j$**，而非底层 PPO 或 reward 公式本身。

**对 wiki 的映射：** [legged_gym](../../wiki/entities/legged-gym.md) 手工地形课程 vs 本文 **LP 自动课程** 选型对比。

## BibTeX（arXiv）

```bibtex
@article{li2026lpacrl,
  title={Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning},
  author={Li, Ziming and Li, Chenhao and Hutter, Marco},
  journal={arXiv preprint arXiv:2601.17428},
  year={2026}
}
```
