# TEMPO（arXiv:2608.07314）

> 来源归档（ingest）

- **标题：** TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models
- **缩写：** **TEMPO**（Two-timescale sEMantic-action decouPled RL pOst-training）
- **类型：** paper / vla / rl-post-training / td3 / calvin / manipulation
- **arXiv：** <https://arxiv.org/abs/2608.07314>
- **HTML：** <https://arxiv.org/html/2608.07314>
- **PDF：** <https://arxiv.org/pdf/2608.07314>
- **项目页：** <https://anonymous.4open.science/w/tempo-page/> — 归档见 [`sources/sites/tempo-anonymous-4open.md`](../sites/tempo-anonymous-4open.md)
- **代码：** 截至 **2026-08-11** 论文与匿名项目页 **未列可运行训练仓**（页面临 Cloudflare 挑战，未见 Code URL）
- **作者：** Ziheng Liu（浙江工商大学）、Quantao Yang\*（KTH；通讯 quantao@kth.se）
- **机构：** 浙江工商大学计算机学院；KTH Robotics, Perception, and Learning
- **入库日期：** 2026-08-11
- **一句话说明：** 在 **FLOWER** VLA 上做 **语义–动作解耦、双频 TD3** 后训练：冻结 VLM，慢更 semantic projection、快更 action expert；CALVIN ABC→D **SR5 81.7% / Avg.Len. 4.59**，真机两任务奖励高于单环 FLOWER-RL。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2608.07314>
- **核心贡献：** SFT 受分布失配；现有 VLA RL 后训练常对所有可训模块用 **统一更新策略**。TEMPO 冻结预训练 VLM，对 **semantic projection** 与 **action expert** 各开一条 TD3 环，并设 **动作侧更高更新频率**，稳住 latent action、快速吸收控制反馈。
- **对 wiki 的映射：**
  - [TEMPO 实体](../../wiki/entities/paper-tempo.md)
  - [VLA](../../wiki/methods/vla.md)
  - [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)

### 2) 双环 TD3 与频率比 \(\rho=f_a:f_s\)（§III）

- **链接：** Method
- **核心贡献：**
  - 实例化 FLOWER：\(h_t=\mathrm{VL}(o_t,l_t)\)，\(z_t=\pi_\theta^s(h_t)\)，\(\mathbf{a}_t=\pi_\phi^a(z_t)\)（action chunk）。
  - 语义环状态 \(h\)、动作 \(z\)；动作环状态 \(z\)、动作 chunk \(\mathbf{a}\)；共享稀疏终局奖励 \(r\in\{0,1\}\)。
  - 各自 replay / twin critic / target；梯度不回传进冻结 VLM，两环之间不直接耦合梯度。
  - 默认强调 \(f_a>f_s\)（文中 **5:1 / 10:1** 有效；**1:1** 几乎无增益）。
- **对 wiki 的映射：**
  - [TEMPO 实体](../../wiki/entities/paper-tempo.md) — 流程总览
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Action Chunking](../../wiki/methods/action-chunking.md)

### 3) CALVIN 与真机（§IV）

- **链接：** Experiments
- **核心贡献：**
  - CALVIN ABC→D：TEMPO **SR5 81.7%**、Avg.Len. **4.59**；FLOWER 77.8%/4.49；FLOWER-RL 78.4%/4.51；相对最强基线 DeFI（81.2%）略高。
  - 组件消融：只更 projection 或只更 expert 都优于 FLOWER，但不如双环全量。
  - 频率：5:1 与 10:1 抬升 SR5；1:1 双环甚至略差于 FLOWER。
  - 真机：两多阶段操作（抽屉未充分打开时需先开抽屉再操作）；TEMPO 后期评测奖励高于 FLOWER-RL，定性更能学到「先开抽屉」隐式语义。
- **对 wiki 的映射：**
  - [CALVIN](../../wiki/entities/calvin-benchmark.md)
  - [VLA 纵深 Stage 5](../../roadmap/depth-vla.md)
  - [DeFI](../../wiki/methods/defi-decoupled-dynamics-vla.md) — 表中对照基线

### 4) 开源边界（步骤 2.5）

- **链接：** <https://anonymous.4open.science/w/tempo-page/>
- **核心贡献：** 匿名评审项目页；抓取遇 Cloudflare「Just a moment…」。论文未给 GitHub。截至入库日按 **确认未开源（匿名页无可核验代码链接）**。
- **对 wiki 的映射：**
  - [项目页归档](../sites/tempo-anonymous-4open.md)
  - [TEMPO 实体](../../wiki/entities/paper-tempo.md) — 源码运行时序图不适用

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-tempo.md`](../../wiki/entities/paper-tempo.md)
- 项目页归档：[`sources/sites/tempo-anonymous-4open.md`](../sites/tempo-anonymous-4open.md)
- 互链参考：[VLA](../../wiki/methods/vla.md)、[CALVIN](../../wiki/entities/calvin-benchmark.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[VLA 纵深](../../roadmap/depth-vla.md)、[VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)、[DeFI](../../wiki/methods/defi-decoupled-dynamics-vla.md)

## BibTeX（arXiv）

```bibtex
@misc{liu2026tempo,
  title={TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models},
  author={Liu, Ziheng and Yang, Quantao},
  year={2026},
  eprint={2608.07314},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
