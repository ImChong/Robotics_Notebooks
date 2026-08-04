# Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?（CoRL 2026）

> 来源归档（ingest · 项目页 PDF + 站点摘要）

- **标题：** Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?
- **类型：** paper / imitation-learning / action-chunking / behavior-cloning / analysis
- **会议：** CoRL 2026（PDF 元数据；项目页 BibTeX 写作 `@article{lazzati2026chunking}`）
- **项目页：** <https://action-chunking.github.io/> — 归档见 [`sources/sites/action-chunking-github-io.md`](../sites/action-chunking-github-io.md)
- **PDF：** <https://action-chunking.github.io/static/action_chunking.pdf>
- **arXiv：** 入库时 **暂无编号**（页上写 Coming soon）
- **代码：** 入库时 **Coming soon**（无可用 GitHub URL）
- **作者：** Filippo Lazzati、Kyle Stachowicz、William Chen、Alberto Maria Metelli、Andrew Wagenmaker、Sergey Levine
- **机构：** 米兰理工大学（Politecnico di Milano）；加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-08-04
- **一句话说明：** 系统消融「为何 action chunking 能抬高 BC」——否定时序一致性 / 有效地平线 / 表征学习作为充分解释；提出 **delayed policy**（\(a_t\mid o_{t-n}\)）与 **隐式集成** 两条主因，并用 **Randomized Delay Ensemble（RDE）** 在仿真与 Franka 真机上匹配标准 chunk 执行。

## 开源状态（步骤 2.5）

- **核查日：** 2026-08-04，打开 <https://action-chunking.github.io/>。
- **已发布：** 项目页、PDF、presentation。
- **未发布：** arXiv 链接、代码仓库。
- **结论：** **宣称将开源 / 待发布**。wiki 实体页不得写「已开源」；`## 源码运行时序图` 写 **不适用**。

## 摘录 1：否定既有假说，提出三条机制

既有常见解释——**temporal consistency**（chunk 内动作联合平滑）、**horizon reduction**（每 \(k\) 步才重决策）、**representation learning**（训练长 chunk 即便只执行前几步也有表征红利）——在 LIBERO / Robomimic 消融下均不足以单独解释成功。论文主张收益主要来自：

1. **非马尔可夫表达力：** 人类演示常含暂停 / 决策边界等非马尔可夫行为；用过去观测预测当前动作即可捕捉，不必维持 chunk 联合分布。
2. **降低复合误差：** Markov BC 存在 \(\Omega(2^H\epsilon)\) 下界；chunk / delay 共享 \(\mathcal{O}((k+1)^{H/k}\epsilon)\) 上界，且该上界对 chunk **紧**——关键不是「少决策」，而是 **平均在更早、误差更小的状态上做条件化**。
3. **隐式集成：** 训练 chunk 同时学习 \(a_t\mid o_t,\ldots,a_t\mid o_{t-k+1}\)，部署时环境交互等价于随时间聚合多条时延关系，带来 ensemble-like 泛化。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-why-action-chunking-improves-bc.md`](../../wiki/entities/paper-why-action-chunking-improves-bc.md)；回写 [`wiki/methods/action-chunking.md`](../../wiki/methods/action-chunking.md) 的「常见误区 / 机制」与 [`wiki/methods/behavior-cloning.md`](../../wiki/methods/behavior-cloning.md)。

## 摘录 2：Delayed policy 与 Randomized Delay Ensemble（RDE）

- **Delayed policy：** 每步仍输出单动作，但条件于 \(o_{t-n}\)：\(\pi_{\mathrm{delay}}^n[\hat\pi_k]\)。在 LIBERO-90 上，合适延迟可 **匹配或超过** 标准 action chunking（Table 1：Delay 94.0% vs AC 89.2%，Markov 68.9%）。
- **RDE：** 每步采样 \(i\sim\mathrm{unif}([n])\)，执行 \([\hat\pi_k(o_{t-i})]_i\)——在**不播放整段 chunk**的情况下同时利用「过去观测条件化」与「多种时延关系」。Table 1 中 RDE 在多数设定贴近 AC；Tool Hang 上 71.8% vs AC 75.2%（Delay 仅 51.6%）。
- **Ordered 对照（Appendix B.1）：** 按固定顺序轮转延迟策略（`AC(10)-Ordered`）在 LIBERO-90 上甚至 93.5% > AC 89.2%，进一步说明 **联合时序一致性并非必需**。

**对 wiki 的映射：** 实体页画「训练 chunk → 部署 AC / Delay / RDE / 显式 Ens」流程图；方法页补充「chunk 训练 ≠ 必须 chunk 执行」。

## 摘录 3：主结果数字（Table 1 / Table 2）与真机

**Table 1（成功率 %，均值±SEM；AC / Delay 取文中报告的 best-case 口径）：**

| Task | Markovian | AC(n) | Delay(n) | AC(n)-RDE | AC(n)-TE |
|------|-----------|-------|----------|-----------|----------|
| Libero-90 | 68.9±0.6 | 89.2±0.2 | **94.0±0.1** | 93.6±0.1 | 92.7±0.1 |
| Libero-10 | 19.8±1.5 | 88.7±0.4 | 86.8±0.4 | 88.5±0.4 | 86.0±0.5 |
| Robomimic Can PH | 83.7±0.4 | 97.2±0.3 | 93.5±0.4 | 96.7±0.2 | 96.2±0.3 |
| Robomimic Square PH | 69.0±0.8 | 85.4±0.5 | 80.8±0.6 | 82.4±0.6 | 80.6±0.5 |
| Robomimic Transport PH | 3.3±0.3 | 12.6±0.5 | 7.9±0.5 | 12.1±0.5 | 12.2±0.6 |
| Robomimic Tool Hang PH | 28.0±0.8 | 75.2±0.5 | 51.6±0.9 | 71.8±0.8 | 42.2±1.0 |

**Table 2（显式集成进一步抬升）：** 如 Robomimic Transport 上 AC 12.6% → AC-Ens **41.5%**；Tool Hang AC 75.2% → AC-Ens random **87.6%**。读点：隐式集成解释「为何单 Delay 不够」，显式集成则是可操作的下一步。

**真机（Franka，15 Hz，delta joint，diffusion policy 训练 \(k=20\)）：** 50 demos / 任务，50 rollouts；Delay 大幅好于 Markov；**RDE 平均匹配并略超标准 AC**。真机部署额外固定 Delay(2)（用 \(o_{t-1}\)）以便异步推理。

**策略族：** 主实验为自训 **diffusion policy（DDPM）**；附录 B.5 用 openpi **π₀.₅** Libero 微调权重做兼容性核对。

**对 wiki 的映射：** 与 [LIBERO](../../wiki/entities/libero-benchmark.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md)、[Action Chunking](../../wiki/methods/action-chunking.md) 互链。

## 摘录 4：工程可读 takeaways

1. **先试 delayed BC**：在只需抬成功率、且可接受反应滞后时，\(a_t\mid o_{t-n}\) 往往比盲目加长 chunk 更干净。
2. **已有 chunk 策略可改部署**：同一 \(\hat\pi_k\) 用 RDE 重部署，多数设定不必坚持「执行整段 chunk」。
3. **真要超 AC：显式集成延迟策略**（多 seed / 多模型），代价是推理算力。
4. **不要把 ACT 式 temporal ensemble 与本文 RDE 混为一谈**：ACT 指数加权会压近时刻贡献；本文线性 TE 与 **随机延迟** 的收益机制不同。

## BibTeX（项目页）

```bibtex
@article{lazzati2026chunking,
  author  = {Filippo Lazzati and Kyle Stachowicz and William Chen and Alberto Maria Metelli and Andrew Wagenmaker and Sergey Levine},
  title   = {Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?},
  year    = {2026},
}
```
