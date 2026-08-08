# PFM-HR: Pose Flow Matching for Humanoid Robots（arXiv:2608.03227）

> 来源归档（ingest）

- **标题：** PFM-HR: Pose Flow Matching for Humanoid Robots
- **缩写 / 框架：** **PFM-HR**（**P**ose **F**low **M**atching for **H**umanoid **R**obots）；核心分数 **PGS**（**P**ose **G**eometry **S**core）
- **类型：** paper / humanoid / motion-tracking / flow-matching / pose-prior / reinforcement-learning
- **arXiv：** <https://arxiv.org/abs/2608.03227>（Submitted 2026-08-04；PDF：<https://arxiv.org/pdf/2608.03227>；HTML：<https://arxiv.org/html/2608.03227v1>）
- **项目页：** <https://gaoyukang33.github.io/PFM-HR.web/>（归档见 [`sources/sites/pfm-hr-web.md`](../sites/pfm-hr-web.md)）
- **代码：** <https://github.com/gaoyukang33/PFM-HR>（**Coming Soon** 占位；归档见 [`sources/repos/pfm-hr.md`](../repos/pfm-hr.md)）
- **作者：** Yukang Gao\*、Yi Gu\*、Yangchen Zhou\*、Xingyu Chen\*、Zhaorui Wang、Fanghai Zhang、Hanyang Cao、Zhengyang Shen、Ji Ma、Runhan Zhang、Lei Han、Renjing Xu†（\* equal contribution；† corresponding）
- **机构：** 香港科技大学广州校区（HKUST-GZ）；诺亦腾机器人（Noitom Robotics）；清华大学深圳国际研究生院（SIGS, Tsinghua）；谷歌（Google）
- **入库日期：** 2026-08-08
- **一句话说明：** 在大规模**无序姿态**上预训练可复用 Flow Matching 姿态先验，用 **Pose Geometry Score（PGS）** 衡量 rollout 关节变化是否对齐先验局部几何，并以此调制跟踪奖励；冻结先验可挂到 ADD / BeyondMimic，提升高动态单轨迹与通用运动跟踪样本效率。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-08）：** [gaoyukang33.github.io/PFM-HR.web](https://gaoyukang33.github.io/PFM-HR.web/) 顶部提供 Paper / arXiv / **Code** 按钮；Code 指向 <https://github.com/gaoyukang33/PFM-HR>。页内展示单轨迹（Backflip、Double Kong 等）、通用运动跟踪与 BeyondMimic 真机部署对比视频。
- **仓库核查（2026-08-08）：** [gaoyukang33/PFM-HR](https://github.com/gaoyukang33/PFM-HR) 仅含 `LICENSE`（MIT）与 README（正文 `# PFM-HR` + `Coming Soon ！！！`）；**无可辨识训练 / 推理脚本、权重或数据入口**。
- **结论：** **宣称开源 / 实现待发布**。wiki 不得写「已可复现训练」；源码运行时序图标 **不适用**，待正式 release 后再补。

## 摘录 1：问题与主张（§I / Abstract）

- **痛点：** 运动先验能改善物理人形跟踪，但 **时序先验**（如 SMP）需要有序 clip 且在线推理重；**姿态先验**（如 PDF-HR）通常只打分单个姿态，不显式评估策略引起的关节协同变化。
- **主张：** **PFM-HR** 在无序姿态集合上训 Flow Matching 去噪器；用去噪映射 Jacobian 对归一化关节差分的响应定义 **PGS**，经参考轨迹校准后调制跟踪奖励 \(r^{T}\)，先验在策略学习中保持冻结。
- **规模：** 演示在 BONES-SEED 上预训练至 **6000 万** 姿态；可继续训练扩展（30M→60M 约 65 GPU-h vs 从头 100 GPU-h）；同规模 PDF-HR 距离监督构造在作者实现中 >500 GPU-h（不含网络训练）。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-pfm-hr.md`](../../wiki/entities/paper-pfm-hr.md)；与 [ADD](../../wiki/methods/add.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[PDF-HR 索引页](../../wiki/entities/paper-notebook-pdf-hr.md)、[SMP](../../wiki/methods/smp.md)、[MimicKit](../../wiki/entities/mimickit.md) 互链。

## 摘录 2：方法栈（§III）

| 模块 | 要点 |
|------|------|
| **姿态表示** | 驱动关节配置 \(\mathbf{q}\in\mathbb{R}^{N_J}\)（文中 G1 \(N_J=29\)）；归一化后作先验输入 |
| **Flow Matching 先验** | 线性路径 \(\boldsymbol{z}_t=t\boldsymbol{x}+(1-t)\boldsymbol{\epsilon}\)；JiT 式 clean-pose 预测 \(F_\phi\)；无序姿态边际分布，无需距离对或时序窗 |
| **PGS** | \(\boldsymbol{d}_k\) 为归一化关节差分方向；\(s_{\mathrm{PGS}}=\|J_\phi\boldsymbol{d}_k\|_2^2\)（JVP，不显式建 Jacobian）；人口最优下与条件协方差二次型对齐 |
| **\(t_{\mathrm{eval}}\)** | 通用跟踪消融取 **0.75**（过小欠特异、近 1 近恒等） |
| **奖励调制** | 参考轨迹上建 PGS 经验 CDF；rollout 分位映射 \(\rho\)，\(r^{P}=\exp(-\alpha\rho)\)，\(r=w^{G}r^{G}+w^{T}r^{T}r^{P}\)（\(\alpha=0.5\)，\(p_{\mathrm{good}}=0.05\)，\(p_{\mathrm{bad}}=0.01\)） |

**对 wiki 的映射：** 实体页画「无序姿态 → FM 先验 → PGS → 调制跟踪奖励 → 冻结挂载 ADD/BM」流程图；强调相对 PDF-HR（姿态距离）与 SMP（时序 score / SDS）的定位。

## 摘录 3：实验与真机（§IV）

| 设定 | 读点 |
|------|------|
| **骨干** | 同 ADD 跟踪骨干；对照 vanilla ADD、ADD w/ PDF-HR、ADD w/ PFM-HR；先验同用未镜像 BONES-SEED |
| **单轨迹（MimicKit 9 技能）** | Backflip / Double Kong 上 ADD 失败；PFM-HR 相对 PDF-HR 更少样本、更低位置误差；高动态技能上样本效率优势更明显 |
| **通用跟踪（LaFAN1 34 段）** | 10/20/30 s 水平均位置误差相对 ADD **−7.6%**、相对 PDF-HR **−10.3%** |
| **BeyondMimic 部署** | 仅在仿真训练挂冻结先验；相对原 BM，达 \(SR\geq80\%\) 样本：Spinkick **−24.2%**、Kick combo **−15.1%**（页叙述）；Table I 四技能均最少样本 |
| **消融** | 先验数据 15M/30M/60M 规模↑则跟踪↑；x-pred 优于 v-pred（尤其 Backflip）；PGS 优于同先验的 FM-Recon（多噪声重建 / SDS 风格） |
| **开销** | batch 4096 / RTX 4090：三水平 denoiser 重建 **1.8 ms** vs 一次 JVP **0.75 ms** |

**对 wiki 的映射：** 「结论」写清：真影响是高动态样本效率与冻结可插拔；代价是无时序方向敏感性与代码尚未发布。

## 局限（§VI）

- PGS 只反映边际姿态局部共变，**不**评估时序方向 / 顺序；反向差分可能得相近分数。
- 依赖姿态语料覆盖；分布外姿态引导变弱。
- 作者展望：符号敏感分数与时序条件先验，同时保留效率与可复用性。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-pfm-hr.md`**（含流程总览；源码时序图标不适用）。
- 新建 **`sources/sites/pfm-hr-web.md`**、**`sources/repos/pfm-hr.md`**。
- 交叉更新：[BeyondMimic](../../wiki/methods/beyondmimic.md)、[ADD](../../wiki/methods/add.md)、[PDF-HR](../../wiki/entities/paper-notebook-pdf-hr.md)、[SMP](../../wiki/methods/smp.md)、[运动跟踪选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)、[AMP/ADD/SMP 对比](../../wiki/comparisons/amp-add-smp-motion-prior-variants.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（Coming Soon）
- [ ] 官方代码正式发布后补源码运行时序图
