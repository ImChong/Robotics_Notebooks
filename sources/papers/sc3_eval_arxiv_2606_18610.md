# SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation（arXiv:2606.18610）

> 来源归档（ingest）

- **标题：** SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation
- **类型：** paper / policy evaluation / video world model / self-consistency
- **arXiv：** <https://arxiv.org/abs/2606.18610>（PDF：<https://arxiv.org/pdf/2606.18610.pdf>）
- **项目页：** <https://weichengtseng.github.io/sc3-eval/>
- **代码：** 确认未开源（项目页无 Code；`WeiChengTseng/sc3-eval` 仅为静态页）
- **作者：** Wei-Cheng Tseng、Gashon Hussein、Yuzhu Dong、Allen Z. Ren、Lucy X. Shi、XuDong Wang、Sergey Levine、Zhaoshuo Li、Jinwei Gu、Florian Shkurti、Ming-Yu Liu、Quan Vuong
- **机构：** 多伦多大学（University of Toronto）、矢量研究所（Vector Institute）、英伟达（NVIDIA）、物理智能（Physical Intelligence）、斯坦福大学（Stanford）、加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-08-11
- **一句话说明：** 把预训练视频基础模型（Cosmos3-Nano + 统一动力学骨干）改造成 **自一致** 策略评估器：联合训练前向/逆向动力学与跨视角 inpainting，并在推理时用逆动力学一致性误差做 early termination；七个真机 π₀.₅ checkpoint 上闭环 Pearson **0.929**、MMRV **0.119**，优于 Ctrl-World / IRASim / Cosmos-Predict 2.5。

## 开源状态（项目页核查，2026-08-11）

- **确认未开源：** 项目页仅 Paper 链；无训练/推理代码、权重或公开数据集下载入口。GitHub `WeiChengTseng/sc3-eval` 为 Pages 静态站，不构成可复现实现。

## 摘要级要点

- **动机：** 真机评测贵且慢；动作条件视频 WM 可代理 rollout，但面临 **自回归漂移**、**多相机互不一致**、以及 **策略行为 OOD** 三难。
- **三一致性（SC3）：** (1) **forward–inverse dynamics** — 共享参数联合训前向帧预测与逆动力学动作恢复，把生成锚定到可解释动作流形；(2) **cross-view consistency** — 随机遮住一视角做 inpainting，无需显式 memory；(3) **test-time consistency** — 复用逆动力学模式算 \(U_{\mathrm{chunk}}\)，超阈值 \(\tau\) 即终止不可靠想象。
- **数据 / 骨干：** 自采 **381 h** 真机 table bussing（12 类物体，三同步相机）；初始化 **Cosmos3-Nano**；flow matching；训练 **32×GB200** ≈ **2.2** 天；闭环推理约 **2.3 s / chunk**（单 GB200）。
- **评测：** 七个 π₀.₅ checkpoint；InD table bussing + OOD reverse bussing；offline / online；人类盲评 language following / lifting / placing；并报告 failure-mode 复现率。

## 核心论文摘录（MVP）

### 1) 自一致训练三模式

- **链接：** §3.2；Fig. 2
- **摘录要点：** 同一 transformer 上随机采样 FD / CVI / ID（\(p_{\mathrm{FD}}=0.8,\ p_{\mathrm{CVI}}=0.1,\ p_{\mathrm{ID}}=0.1\)）；前向从动作去噪视频、跨视角补全、逆动力学从视频去噪动作 chunk；共享权重使前向被逆向可恢复性隐式正则化。
- **对 wiki 的映射：**
  - [SC3-Eval](../../wiki/entities/paper-sc3-eval.md) — 核心机制。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 视频 WM 评估谱系。

### 2) 闭环 rollout + 不确定性早停

- **链接：** §3.3；Alg. 1
- **摘录要点：** 策略出 \(l'\) 步动作 → 前向生成 → 只保留前 \(l\) 帧进入下一观测（prediction–execution horizon decoupling）；\(U_{\mathrm{chunk}}=\frac{1}{l}\sum\|a_i-\hat a_i\|_2\)，\(>\tau\) 则 break。
- **对 wiki 的映射：**
  - [SC3-Eval](../../wiki/entities/paper-sc3-eval.md) — 推理协议。
  - [评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) — WM 作策略评估器。

### 3) 与真机相关及 failure-mode 复现

- **链接：** §4.2–4.4；Tab. 1–2；Fig. 4–6
- **摘录要点：** 全量闭环 \(r=0.929\)、MMRV \(0.119\)；InD online \(r=0.984\)；消融去 ID / 去 CVI / 去 early-term / 去 horizon decoupling 均掉点；相对基线更能复现 language/lifting/placing 失败类别。
- **对 wiki 的映射：**
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md)、[IRASim](../../wiki/entities/paper-irasim.md)、[GigaWorld-1](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md) — 同族对照。

## BibTeX

```bibtex
@article{tseng2026sc3eval,
  title   = {SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation},
  author  = {Tseng, Wei-Cheng and Hussein, Gashon and Dong, Yuzhu and Ren, Allen Z. and Shi, Lucy X. and Wang, XuDong and Levine, Sergey and Li, Zhaoshuo and Gu, Jinwei and Shkurti, Florian and Liu, Ming-Yu and Vuong, Quan},
  journal = {arXiv preprint arXiv:2606.18610},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-sc3-eval.md`](../../wiki/entities/paper-sc3-eval.md)
- 项目页：[`sources/sites/weichengtseng-sc3-eval.md`](../sites/weichengtseng-sc3-eval.md)
- 策展索引源（升格前）：[`sources/papers/sun_awesome_wm_2606_18610_sc3-eval-evaluating-robot-foundation-mod.md`](./sun_awesome_wm_2606_18610_sc3-eval-evaluating-robot-foundation-mod.md)
- 互链：[Ctrl-World](../../wiki/entities/paper-ctrl-world.md)、[IRASim](../../wiki/entities/paper-irasim.md)、[GigaWorld-1](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md)、[评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)
