# ParkourFormer: Integrating Predictive Supervision and Sequence Modeling into Parkour Locomotion（arXiv:2605.25782）

> 来源归档（ingest）

- **标题：** ParkourFormer: Integrating Predictive Supervision and Sequence Modeling into Parkour Locomotion
- **缩写：** **ParkourFormer**
- **类型：** paper / humanoid / parkour / sequence-modeling / future-prediction / amp / rgb-d
- **来源：** [arXiv:2605.25782](https://arxiv.org/abs/2605.25782)（HTML：[ar5iv](https://ar5iv.labs.arxiv.org/html/2605.25782)）
- **项目页：** <https://mronaldo-gif.github.io/parkourformer.github.io/> — 归档见 [`sources/sites/parkourformer-github-io.md`](../sites/parkourformer-github-io.md)
- **PDF：** <https://arxiv.org/pdf/2605.25782>
- **作者：** Yanheng Mai、Wenhao Xu、Zirui Huang、Yifei Fu、Shengwei Dong、Xinjue Wang、Kailun Huang、Yanzhe Xie、Renjing Xu†（† corresponding，`renjingxu@hkust-gz.edu.cn`）
- **机构：** 香港科技大学广州校区（HKUST-GZ）；CLAI-LAB / CL-TECH；华南农业大学（SCAU）；广东工业大学（GDUT）
- **发表：** arXiv preprint，2026（检索到 v3，2026-06-12）
- **入库日期：** 2026-08-16
- **最后更新：** 2026-08-16
- **一句话说明：** 把人形跑酷写成 **future-conditioned Seq2Seq**：当前状态用 cross-attention 查询历史，预测头监督未来两步本体/AMP 状态，再把预测未来拼进动作头与 AMP 判别器；G1 上九类地形统一策略平均穿越成功率 **93.85%**。

## 开源状态（步骤 2.5）

- **核查日：** 2026-08-16。打开项目页、作者 GitHub [`MRonaldo-gif`](https://github.com/MRonaldo-gif) 与站点仓 [`parkourformer.github.io`](https://github.com/MRonaldo-gif/parkourformer.github.io)。
- **已发布：** 项目页、arXiv PDF/HTML、真机与仿真演示视频（页内嵌入）。
- **未发布：** 训练/推理代码、权重、数据集。站点仓仅为 github.io 主页；作者公开仓无训练入口；论文未承诺开源。
- **结论：** **确认未开源**。wiki 实体页「源码运行时序图」写 **不适用**。

## 摘录 1：问题与三条贡献

现有人形/足式 RL 策略大多是 **reactive**：观测直接映射动作，未来接触与动量只隐式藏在隐状态里。跑酷（楼梯、缺口、坡、障碍）要求当前指令与即将到来的接触相一致。ParkourFormer 把全身控制改写成 **query-based sequence-to-sequence**：

1. **Query-based Temporal Reasoning：** 「now → past → future」——当前状态主动查询历史轨迹，而不是被动编码。
2. **Future-conditioned Policy Learning：** 显式预测短时域本体状态，并用监督信号把动作条件化到预测未来上。
3. **Unified One-Policy Multi-Terrain：** 单一策略覆盖九类地形，不按地形拆奖励或拆网络。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-parkourformer.md`](../../wiki/entities/paper-parkourformer.md)；对照 [Hiking in the Wild](../../wiki/entities/paper-hiking-in-the-wild.md)（同用 Project Instinct MuJoCo）、[PHP](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[LightLP](../../wiki/entities/paper-light-loco-parkour.md)、[SSR](../../wiki/entities/paper-ssr-humanoid-open-world-traversal.md)、[Next Token Prediction](../../wiki/entities/paper-notebook-humanoid-locomotion-as-next-token-prediction.md)。

## 摘录 2：方法栈（§3）

仿真：**Project Instinct** 的 **MuJoCo** 管线（与 Hiking in the Wild 同族）；**4096** 并行环境；仿真 **200 Hz** / 控制 **50 Hz**；最多 **30,000** iter；单卡 **RTX 4090D**。平台：**Unitree G1，29 DoF**。

| 量 | 定义 |
|----|------|
| \(o_t\in\mathbb{R}^{96}\) | \([\boldsymbol{\omega}_a, \mathbf{g}_p, \mathbf{v}_c, \mathbf{q}_p, \mathbf{q}_v, \mathbf{a}_{t-1}]\) |
| \(\mathbf{d}_t\) | 当前 RGB-D 帧 → 深度 token \(\mathbf{z}_t=\mathcal{F}_{\mathrm{RGB-D}}(\mathbf{d}_t)\in\mathbb{R}^{128}\) |
| \(s_t\in\mathbb{R}^{67}\) | AMP 风格运动状态 \([\mathbf{g}_p, \mathbf{q}_p, \mathbf{q}_v, \mathbf{v}_l, \boldsymbol{\omega}_a]\) |
| 历史 | \(\mathbf{o}_t=\{o_{t-7},\ldots,o_t\}\)（8 帧） |
| 动作 | \(\mathbf{a}_t\in\mathbb{R}^{29}\)，名义姿态上的 action delta，底层 PD |

骨干：当前观测 + 深度 token 作 **query**，历史 token 作 **key/value**，多层 cross-attention + residual FFN。地形上下文 \(c_t\) 经 **Conditional SwiGLU** 乘性门控调制中间特征。非对称 critic 额外吃特权线速度 \(\mathbf{v}_l\)。

**未来预测头：** 从 Transformer 特征确定性预测未来 **两步** AMP 状态 \(\hat{\mathbf{s}}_{t+1:t+2}\)。监督：

\[
\mathcal{L}_{\mathrm{pred}}=\frac{1}{|\mathcal{M}_{\mathrm{pred}}|}\sum_i\sum_{k=1}^{2}m_{i,k}\,\|\hat{s}_{i,k}-s_{i,k}\|_2^2
\]

无效 episode 用 mask \(m_{i,k}\) 去掉。预测未来拼到 AMP 历史：\(\tilde{\mathbf{s}}_t=[\mathbf{s}_t;\hat{\mathbf{s}}_{t+1:t+2}]\)（8+2=10 帧），让判别器看到「已观测 + 预期」连续流形。动作头 \(\pi_\theta'\) 条件于 \(\mathbf{Q}_t^{(L)}\) 与 \(\hat{\mathbf{s}}_{t+1:t+2}\)。

总奖励 \(R_{\mathrm{total}}=R_{\mathrm{task}}+R_{\mathrm{AMP}}\)。联合损失：

\[
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{ppo}}+c_1\mathcal{L}_{\mathrm{value}}+c_2\mathcal{L}_{\mathrm{pred}}-c_3\mathcal{H}[\pi]-c_4\mathcal{L}_{\mathrm{AMP}}
\]

\(c_2\) **按 advantage 加权**：负 advantage 样本加大预测损失权重，正 advantage 保持 1.0。

**对 wiki 的映射：** [AMP 奖励](../../wiki/methods/amp-reward.md)、[强化学习](../../wiki/methods/reinforcement-learning.md)、[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)。

## 摘录 3：九类地形与主结果（Table 1–3）

训练地形（Fig. 5）：Boxes、Walk Over Obstacles、Climb Slope、Rough ground、Up Stairs、Climb Down、Down stairs、Climb Up、Gaps Crossing；每类 **L1–L9** 难度。

**Table 1 平均穿越成功率（%）：**

| 模型 | Boxes | Obstacles | Slope | Rough | Up Stairs | Climb Down | Down stairs | Climb Up | Gaps | Mean |
|------|------:|----------:|------:|------:|----------:|-----------:|------------:|---------:|-----:|-----:|
| 1-MLP（无 MoE） | 57.28 | 66.56 | 81.52 | 94.18 | 4.58 | 94.38 | 0.14 | 21.40 | 0.54 | 46.73 |
| 4-MLP（MoE） | 81.54 | 74.98 | 82.30 | 92.38 | 90.62 | 94.24 | 81.02 | 94.58 | 92.74 | 87.16 |
| Vanilla Transformer | 89.86 | 84.50 | 85.68 | 94.22 | 89.98 | 94.68 | 90.44 | 94.12 | 90.96 | 90.49 |
| **ParkourFormer** | **91.18** | **86.12** | **95.32** | **95.18** | **94.98** | **95.24** | **95.42** | **94.98** | **96.20** | **93.85** |

相对 1-MLP 的均值增益约 **+47.12 pt**（摘要「up to 47.12%」即此对照）。Table 2：Target Near Ratio **0.489**、Tracking Vel **0.837**（均高于三基线）。

**Table 3 消融（Mean）：** 全文 **93.85**；去 MSE 监督 **82.87**（下楼塌到 **9.50%**）；去 RGB-D query **80.08**（缺口 **24.24%**、Climb Up **75.84%**）；去未来预测头 **92.79**。读法：容量（Transformer vs MLP）已经把均值抬到 90%；**监督信号**保住下楼，**RGB-D** 保住缺口/攀升，预测头再给约 **+1 pt** 与更稳的优化曲线。

真机：G1 在楼梯、平台、缺口、不规则障碍上展示统一策略；项目页另有 L9 仿真最高难度片段。

**对 wiki 的映射：** [楼梯/障碍感知 locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[感知越障路线](../../roadmap/depth-perceptive-locomotion.md)。

## 摘录 4：局限（§6）

1. 地形更杂时奖励仍稀疏，大缺口/不规则障碍上优化信号弱。
2. AMP 每次 reset 随机切参考起点，**不带地形上下文**、不保跨 reset 连续性，判别器难给地形条件化指导。
3. 重度依赖 RGB-D；传感器损坏或缺失会导致功能整体失效。
4. 传感器建模与接触仿真限制「全难度完美成功」；作者强调对照与消融在同一设定下进行。

## BibTeX（项目页）

```bibtex
@article{mai2026parkourformer,
  title={ParkourFormer: Integrating Predictive Supervision and Sequence Modeling into Parkour Locomotion},
  author={Mai, Yanheng and Xu, Wenhao and Huang, Zirui and Fu, Yifei and Dong, Shengwei and Wang, Xinjue and Huang, Kailun and Xie, Yanzhe and Xu, Renjing},
  journal={arXiv preprint arXiv:2605.25782},
  year={2026}
}
```
