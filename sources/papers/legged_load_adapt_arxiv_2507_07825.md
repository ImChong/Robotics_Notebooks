# Beyond Robustness: Learning Unknown Dynamic Load Adaptation for Quadruped Locomotion on Rough Terrain（arXiv:2507.07825）

> 来源归档（ingest）

- **标题：** Beyond Robustness: Learning Unknown Dynamic Load Adaptation for Quadruped Locomotion on Rough Terrain
- **类型：** paper / quadruped / locomotion / load-adaptation / privileged-training / sim2real
- **arXiv：** <https://arxiv.org/abs/2507.07825>（PDF：<https://arxiv.org/pdf/2507.07825.pdf>）
- **作者：** Leixin Chang、Yuxuan Nai、Hua Chen、Liangjing Yang
- **机构：** 浙江大学国际联合学院（ZJU-UIUC Institute）
- **项目链接：** <https://leixinjonaschang.github.io/leggedloadadapt.github.io/>
- **入库日期：** 2026-08-02
- **一句话说明：** 用 load characteristics（质量/摩擦/位姿/速度）建模箱载动态载荷，经 teacher–student + concurrent load estimator 在盲本体感觉下实现崎岖地形未知动态载荷适应；Unitree Go2 零样本 sim-to-real。

> **URL 校正：** 用户同时给出的 `arxiv.org/abs/2109.12343` 为同名起首 *Beyond Robustness* 的**多机器人韧性综述**（Prorok et al.），与本项目页无关；本 ingest 以项目页 BibTeX / 论文正文为准，采用 **arXiv:2507.07825**。

## 开源状态（核查，2026-08-02）

- **宣称将开源 / 未列可运行仓：** 项目页 Code 按钮文案为 **「Code (comming soon)」**（拼写原文如此），链接回项目页自身；**无独立 GitHub / Hugging Face / Zenodo 训练或部署入口**。
- **可复现边界：** 方法细节（网络维数、奖励、DR 范围、Isaac Gym 8192 并行）已写在论文 §III–IV；真机视频与仿真对比在项目页；**权重与训练代码截至入库日未发布**。
- **源码运行时序图：** wiki 实体页标 **不适用**。

## 摘要级要点

- **问题：** 箱载动态载荷（滑动/滚动）使 CoM 持续漂移；仅靠 domain randomization「当扰动」不够；纯静态载荷辨识忽略机–载互耦。
- **建模：** **Load characteristics** $\boldsymbol{l}_t\in\mathbb{R}^8$ = [位置, 速度, 质量, 摩擦系数]（机体坐标系）。
- **训练：** Teacher–student（特权 latent $z_t$ + 本体历史重建）+ **concurrent load estimator**（监督回归 $\hat{\boldsymbol{l}}_t$）+ 非对称 actor–critic；随后 **student reinforcing**（PPO 微调本体编码器与 actor）。
- **奖励关键项：** load linear velocity $1/(1+\boldsymbol{v}_{\text{load}})$，鼓励载荷相对机体静止；无显式“机身水平”项，平坦姿态为涌现行为。
- **对照：** NLW（无载荷特权/无载荷奖励，鲁棒性基线）、LW（载荷仅进特权支路）、Oracle（actor 直通真值载荷特征）。
- **结果：** 仿真 7 kg / μ≈0.01 崎岖地形上 Ours≈Oracle，显著优于 NLW/LW（LW 在 rough 上摔倒）；Go2 真机 4 kg 铅球穿越软台阶，静止 2/4/6 kg 跌落冲击可适应。

## 核心论文摘录（MVP）

### 1) Load characteristics modeling

- **链接：** §III-A；Oracle 消融 §IV
- **摘录要点：** 用质量、摩擦、位置、速度四类量统一表示箱内动态载荷；作者曾直接建模外力 wrench，效果更差，推测低维结构化特征更易被策略捕捉分布。
- **对 wiki 的映射：**
  - [Legged Load Adapt 实体页](../../wiki/entities/paper-legged-load-adapt-unknown-dynamic-load.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)

### 2) Teacher–student + concurrent estimator + student reinforcing

- **链接：** §III-B；Fig. 2；Table II、IV
- **摘录要点：** 特权编码器压 $s_t,p_t\to z_t$；本体编码器从历史重建 $z^s_t$；载荷估计器并行监督；7500 iter teacher–student 后 1500 iter student reinforce，弥合 $z$ 与 $z^s$ 偏差。
- **对 wiki 的映射：**
  - [Legged Load Adapt 实体页](../../wiki/entities/paper-legged-load-adapt-unknown-dynamic-load.md)
  - [RMA](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 3) 仿真对照与 Go2 sim-to-real

- **链接：** §IV；Fig. 1、5–8；项目页视频
- **摘录要点：** 动态实验（plane/stair/rough/slope）与静止跌落实验；鲁棒性基线（NLW）不足以处理重动态载荷；斜坡优势不显著为作者自承局限。
- **对 wiki 的映射：**
  - [Legged Load Adapt 实体页](../../wiki/entities/paper-legged-load-adapt-unknown-dynamic-load.md)
  - [Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

## BibTeX

```bibtex
@misc{chang2025robustnesslearningunknowndynamic,
  title         = {Beyond Robustness: Learning Unknown Dynamic Load Adaptation for Quadruped Locomotion on Rough Terrain},
  author        = {Leixin Chang and Yuxuan Nai and Hua Chen and Liangjing Yang},
  year          = {2025},
  eprint        = {2507.07825},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2507.07825}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-legged-load-adapt-unknown-dynamic-load.md`](../../wiki/entities/paper-legged-load-adapt-unknown-dynamic-load.md)
- 项目页：[`sources/sites/leggedloadadapt-github-io.md`](../sites/leggedloadadapt-github-io.md)
- 互链：[Privileged Training](../../wiki/concepts/privileged-training.md)、[RMA](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[SplitAdapter](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md)、[Domain Randomization](../../wiki/concepts/domain-randomization.md)
