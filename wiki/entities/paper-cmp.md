---
type: entity
tags: [paper, humanoid, motion-prior, amp, smp, context-aware, rl, hkust-gz, mimickit, unitree-g1]
status: complete
updated: 2026-08-18
arxiv: "2608.03234"
venue: "2026 · arXiv"
related:
  - ./paper-amp-survey-01-amp.md
  - ../methods/amp-reward.md
  - ../methods/smp.md
  - ../methods/ase.md
  - ./paper-bfm-19-calm.md
  - ./paper-bfm-21-case.md
  - ../comparisons/amp-add-smp-motion-prior-variants.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ./mimickit.md
  - ./unitree-g1.md
  - ./paper-pfm-hr.md
  - ./paper-notebook-pdf-hr.md
  - ../queries/humanoid-motion-tracking-method-selection.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
sources:
  - ../../sources/papers/cmp_arxiv_2608_03234.md
summary: "CMP（arXiv:2608.03234，HKUST-GZ）：用高优势 rollout + demo 锚定的对比相关度，把 AMP/SMP 的任务无关先验软重权成上下文条件适配器；五任务提升回报与样本效率，参考失衡更稳；截至 2026-08-18 无官方代码。"
---

# CMP：上下文感知运动先验

**CMP**（*Context-Aware Motion Priors*；论文 *Learning Context-Aware Motion Priors for Humanoid Control*，[arXiv:2608.03234](https://arxiv.org/abs/2608.03234)）由 **香港科技大学广州校区（HKUST-GZ）** 提出：不切数据集、不先做 skill discovery，而是在下游 RL 中学习「任务上下文 ↔ 参考 clip」相关度，软重权参考侧监督，并用轻量残差适配器把 [AMP](../methods/amp-reward.md) / [SMP](../methods/smp.md) 从任务无关先验改成上下文条件先验。同组相邻工作是姿态先验 [PFM-HR](./paper-pfm-hr.md) / [PDF-HR](./paper-notebook-pdf-hr.md)（arXiv:2608.03227 / 2602.04851）。

## 一句话定义

**异构参考库里「看起来像人」≠「对当前目标有用」——用优势信号学相关度，再软重权 AMP/SMP 的参考监督，让先验跟着上下文走。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CMP | Context-Aware Motion Priors | 本文框架：上下文相关度 + 残差适配器 |
| AMP | Adversarial Motion Prior | 主实例化底座：对抗判别风格先验 |
| SMP | Score-Matching Motion Prior | 扩展实例化：冻结扩散 denoiser 先验 |
| GAE | Generalized Advantage Estimation | Online 分支用标准化优势筛正样本 |
| BCE | Binary Cross-Entropy | CMP-AMP 适配器的参考/策略侧分类损失 |

## 为什么重要

- **指出先验盲区：** [AMP](./paper-amp-survey-01-amp.md) / [SMP](../methods/smp.md) 回答「动作是否像参考分布」；CMP 补「在当前目标/命令/物体状态下，哪段参考更该监督」。
- **工程路径轻：** 不引入固定 skill latent、不手工切库；基座先验训练流程不变，只加对比编码器 + 残差适配器。对照 [ASE](../methods/ase.md) / [CALM](./paper-bfm-19-calm.md) / [C·ASE](./paper-bfm-21-case.md) 的「先学技能空间再选」。
- **读数清晰：** 相对 AMP 在五任务上回报与样本效率双升；行走 clip ×100 失衡时 AMP 掉 11.5%、CMP-AMP 仅掉 2.8%；附录 E 在模拟 [G1](./unitree-g1.md) 上复现同趋势。定性上 CMP-AMP 比 AMP 更早到达目标、更早转向加速。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 香港科技大学广州校区（HKUST-GZ）；通讯 Renjing Xu |
| **作者** | Yunyang Mo、Yi Gu、Yangchen Zhou（共同一作）、Hanyang Cao、Renjing Xu |
| **平台** | MimicKit 人形控制任务；附录 E 额外报告模拟 Unitree G1 |
| **数据** | 与 AMP 同款 locomotion 参考集（[MimicKit](./mimickit.md) 分发）；无任务专用 demo / 技能标签 |
| **任务** | Target Location、Steering、Trajectory Following、Dodgeball、Dribbling |
| **栈** | Isaac Gym Preview 4 + PhysX；4096 并行环境；RTX 3090 / 4090 |
| **开源** | **未列可运行入口**（截至 **2026-08-18** abs/HTML 无 GitHub / 项目页；正文称 clip manifest「included in the code」） |

## 核心原理

### 重权视角

任务无关先验的参考项是 \(\mathbb{E}_{x\sim p_E}[\ell^{\mathrm{ref}}(x)]\)。CMP 学 \(R_\phi(c,x)\)，诱导

\[
q_\phi(x\mid c)=w_\phi(c,x)\,p_E(x),\qquad
w_\phi(c,x)\propto\exp(\alpha R_\phi(c,x)).
\]

实践中用 minibatch softmax + clip 得 \(\bar{w}_\phi\)，以 `stop-gradient` 权重训练上下文残差适配器 \(\psi\)；\(\phi\) 与基座 \(\theta\) 不吃适配器梯度。

### 任务上下文 \(c\)（附录 K，局部 heading 系）

| 任务 | 上下文 | 维数 |
|------|--------|------|
| Target Location | 目标平面位置 | \(\mathbb{R}^2\) |
| Steering | 运动方向 + 目标速度 + 朝向 | \(\mathbb{R}^5\) |
| Trajectory Following | 当前及未来路点 \(K=6\)、\(\Delta=0.30\,\mathrm{s}\)（最远 1.50 s） | \(\mathbb{R}^{12}\) |
| Dodgeball | 球相对位置与速度 | \(\mathbb{R}^6\) |
| Dribbling | 球位/速 + 期望方向/速度 | \(\mathbb{R}^9\) |

### 对比相关度（无标签）

余弦相似度 \(R_\phi=f_{\phi_c}(c)^\top g_{\phi_x}(x)\)（\(\ell_2\) 归一）；对比温度 \(\tau=0.1\)。

| 分支 | 正样本 | 作用 |
|------|--------|------|
| Online \(\mathcal{L}_{\mathrm{on}}\) | \(\hat A>0\) 的策略 motion，按优势加权 | 抓「当前任务上有用」的上下文–运动对 |
| Demo \(\mathcal{L}_{\mathrm{demo}}\) | 同批上下文下的参考 clip；rollout 作负 | 把 query 钉在参考支撑上，防漂移 |

正样本过少时取 batch 最高优势子集（最少 64 / fallback 比例 0.35）。消融：去掉 \(\mathcal{L}_{\mathrm{demo}}\) 时检索质量变差（远目标可能漂到库外）；Uniform Adapter / Shuffled Relevance 说明**有适配器容量 ≠ 有正确相关度**。

### AMP / SMP 适配

| 底座 | 适配形式 | 部署期先验奖励 |
|------|----------|----------------|
| AMP | \(l_{\theta,\psi}=\mathrm{sg}[l_\theta]+\lambda_{\mathrm{res}}\Delta l_\psi(c,x)\)，\(\lambda_{\mathrm{res}}=0.03\) | \(-\log(1-\sigma(l_{\theta,\psi}))\) |
| SMP | \(\hat\epsilon=\mathrm{sg}[\epsilon_\theta]+\lambda_{\mathrm{res}}\Delta\epsilon_\psi\)，\(\lambda_{\mathrm{res}}=0.1\)；denoiser 冻结；残差正则 \(\eta=0.001\) | 用适配后误差替代原 SMP reward（\(\mathcal{K}=\{22,15,8\}\)） |

### 流程总览

```mermaid
flowchart TB
  ref["MimicKit locomotion 参考库<br/>p_E(x)"]
  roll["策略 rollout<br/>(c, x_π, Â)"]
  rel["对比相关度 R_φ(c,x)<br/>L_on + λ_demo L_demo"]
  w["软权重 w̄_φ"]
  base["基座先验 θ<br/>AMP 判别器 / SMP denoiser"]
  adapt["上下文残差适配器 ψ"]
  reward["上下文条件先验奖励<br/>+ 任务奖励"]
  ppo["PPO 更新策略"]
  ref --> rel
  roll --> rel
  rel --> w
  ref --> base
  w --> adapt
  base --> adapt
  adapt --> reward --> ppo
  roll --> ppo
```

## 源码运行时序图

**不适用** — 截至复核日（2026-08-18）公开材料**无**官方仓库或项目页 URL；正文提及 clip manifest「included in the code」，但未给出可克隆入口。若后续开源，应按「MimicKit 任务 + AMP/SMP 基座 → 训相关度与适配器 → 五任务评测」补 `sequenceDiagram`。

## 工程实践

| 项 | 做法 |
|----|------|
| 参考库 | 与 AMP 相同 locomotion 集；**不要**为五任务另切技能标签 |
| 运动 clip | MimicKit 10 帧 \(X_t=(x_{t-8},\ldots,x_{t+1})\)；\(\Phi(s)\) 不含任务上下文 |
| 相关度模型 | 上下文 / 运动各 256→128 MLP + SiLU，\(\ell_2\) 归一后点积；\(\tau=0.1\)，\(\alpha=0.5\)（AMP）/ \(1.0\)（SMP） |
| AMP 适配器 | motion/context 双支路 256-d，拼接后融合成标量 \(\Delta l\)；输出层**零初始化**以冷启动对齐基座 |
| 权重稳定 | minibatch 归一化后 clip 到 \([0.5,2.0]\)；适配损失 detach 权重 |
| 读曲线 | 主看 test return（32 episode × 3 seed）+「达基线 80% 最终回报的 env steps」；勿只比最终曲线高度 |
| 失衡场景 | 行走类 clip 过采样时优先上 CMP，而非继续堆均匀 AMP |
| 任务/先验权重 | AMP 与 SMP 均为 \(0.5/0.5\)；CMP-SMP 另有 contrastive/adapter warmup 5/10 iter |

## 实验与评测

**主表（Table 1，mean ± std，三 seed）：** CMP-AMP 五任务回报全面高于 AMP（Steering \(299\pm12\)→\(480\pm6\)、Trajectory \(184\pm1\)→\(302\pm6\)、Dribbling \(319\pm10\)→\(456\pm5\)）；CMP-SMP 增益更小但一致。样本效率（达基线 80% 最终回报的 env steps，×10⁸）：CMP-AMP Location 6.5→2.4、Trajectory 6.8→2.7、Dribbling 5.2→2.8、Dodgeball 7.6→5.6；Steering 与 AMP 持平（均为 1.2）。

**可解释性：** Target Location 上相关度权重随训练从「全局偏行走」演化为「近走、远远跑、侧/后偏转向跑」（约 \(1.3\to5.2\times10^8\) env steps）；带 \(\mathcal{L}_{\mathrm{demo}}\) 时近/远/后目标分别检索到走、跑、制动类 clip。

**消融（Dodgeball / Dribbling）：** Uniform Adapter 与 Shuffled Relevance 无法稳定复现全方法的效率；Online-only 最终回报可接近（Dodgeball 同为 458），但达阈更慢（5.6 vs 6.5）。CMP-AMP 两边达阈样本最少。

**失衡（Table 2，Location）：** 行走 clip ×2 / ×5 / ×20 / ×100 时 AMP 掉 3.2% / 6.4% / 11.5% / 11.5%；CMP-AMP 仅掉 1.3% / 1.1% / 1.9% / 2.8%。

**G1（Appendix E）：** 模拟 G1 上 CMP-AMP 五任务回报与达阈样本全面优于 AMP（Dribbling 294→467，达阈 10.1→2.9×10⁸；Steering 375→480）。

## 结论

**在异构运动先验上，上下文相关度重权比「换一个更大的任务无关先验」更能同时抬回报与样本效率；demo 锚定防止相关度漂出参考支撑，是稳训的第二杠杆。**

1. **先验选型：** 已有 AMP/SMP 基线、任务上下文变化大（目标/命令/物体）时，优先试 CMP，而不是先上 ASE/CALM 式独立 skill 空间。
2. **读主指标：** 同时看最终 return 与「达基线 80% 的 samples」；Steering 上回报大涨但达阈样本持平，说明增益形态因任务而异。
3. **失衡库：** 行走过采样会明显伤 AMP（×20 已掉 11.5%）；CMP 把掉点压到约 3% 内——数据清洗前可先当缓解手段。
4. **消融优先级：** 学对相关度 ≫ 只加上下文适配器容量；\(\mathcal{L}_{\mathrm{demo}}\) 主保检索可信与样本效率。
5. **底座差异：** AMP 上增益更大；SMP 上一致但更小——适配器收益依赖基座如何把监督变成策略梯度。
6. **部署边界：** 全文为仿真；无真机、无运动质量独立评分；复现前先核是否放出代码与 clip manifest。

## 与其他工作对比

| 维度 | CMP | ASE / CALM / C·ASE | 标准 AMP / SMP | PDF-HR / PFM-HR |
|------|-----|--------------------|----------------|-----------------|
| 多样性建模 | 原参考空间软重权 | 先学 latent skill 再选 | 整库均匀先验 | 姿态距离 / 流匹配几何 |
| 何时适配 | 下游 RL 在线 | 常先 skill 阶段再冻结 | 不按任务上下文改参考权重 | 冻结先验调制跟踪奖励 |
| 标签需求 | 无技能标签 | 常需结构/条件化训练 | 无 | 无序姿态语料 |
| 侵入性 | 轻量适配器，基座目标保留 | 引入中间技能接口 | 基线 | 旁挂奖励项 |
| 代码（本库核查） | **无 URL** | 各有公开实现 | MimicKit / 社区复现 | PFM-HR Coming Soon |

同组分工：CMP 改的是 **clip 级参考监督在当前任务上下文下该听谁**；[PDF-HR](./paper-notebook-pdf-hr.md) / [PFM-HR](./paper-pfm-hr.md) 改的是 **姿态是否落在合理流形**。前者挂 AMP/SMP 风格先验，后者挂跟踪奖励。

## 局限与风险

- 参考库**支撑外**的行为无法「变出来」；相关度只重权已有 clip。
- 依赖优势估计：critic 偏、探索不足会污染 online 正样本。
- 评测限于结构化上下文与仿真人形；高维感知、真机、独立运动质量指标未做。
- SMP 增益小于 AMP，说明方法有效性绑定底座表征方式。
- 开源缺口：选型复现前先核实仓库是否上线。

## 关联页面

- [AMP（论文实体）](./paper-amp-survey-01-amp.md) — 任务无关对抗先验源流
- [AMP 方法页](../methods/amp-reward.md) — 风格奖励工程读法
- [SMP](../methods/smp.md) — CMP 的第二实例化底座
- [ASE](../methods/ase.md) — 先学技能空间再选的对照路线
- [CALM](./paper-bfm-19-calm.md) / [C·ASE](./paper-bfm-21-case.md) — 条件 latent skill 对照
- [AMP / ADD / SMP 对比](../comparisons/amp-add-smp-motion-prior-variants.md) — 变体选型表；CMP 可视为「上下文适配层」
- [人形 AMP 综述坐标](../overview/humanoid-amp-motion-prior-survey.md) — 分布约束线扩展阅读
- [MimicKit](./mimickit.md) — 参考数据与实验栈入口
- [Unitree G1](./unitree-g1.md) — 附录 E 形态验证平台
- [PFM-HR](./paper-pfm-hr.md) / [PDF-HR](./paper-notebook-pdf-hr.md) — 同组姿态先验（相邻 arXiv）
- [人形运动跟踪方法选型](../queries/humanoid-motion-tracking-method-selection.md) — 先验路线选型轴

## 参考来源

- [sources/papers/cmp_arxiv_2608_03234.md](../../sources/papers/cmp_arxiv_2608_03234.md) — 本次 ingest 归档
- [arXiv:2608.03234](https://arxiv.org/abs/2608.03234) — 论文与附录

## 推荐继续阅读

- Peng et al., *AMP: Adversarial Motion Priors* ([arXiv:2104.02180](https://arxiv.org/abs/2104.02180))
- Mu et al., *SMP: Reusable Score-Matching Motion Priors* ([arXiv:2512.03028](https://arxiv.org/abs/2512.03028))
- Peng, *MimicKit* ([arXiv:2510.13794](https://arxiv.org/abs/2510.13794))
- Peng et al., *ASE* ([arXiv:2205.01906](https://arxiv.org/abs/2205.01906))
