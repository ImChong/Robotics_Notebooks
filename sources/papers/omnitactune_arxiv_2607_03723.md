# OmniTacTune: Policy-Agnostic Real-World RL for Tactile Residual Adaptation of Visual Policies（arXiv:2607.03723）

> 来源归档（ingest）

- **标题：** OmniTacTune: Policy-Agnostic Real-World RL for Tactile Residual Adaptation of Visual Policies
- **类型：** paper / manipulation / tactile-sensing / real-world RL / residual policy / visuo-tactile adaptation
- **arXiv abs：** <https://arxiv.org/abs/2607.03723>
- **arXiv HTML：** <https://arxiv.org/html/2607.03723v1>
- **PDF：** <https://arxiv.org/pdf/2607.03723>
- **项目页：** <https://colinyu1.github.io/omnitactune-site/>
- **机构：** 马里兰大学帕克分校（University of Maryland, College Park）、佐治亚理工学院（Georgia Institute of Technology）
- **硬件：** xArm7 + 夹爪 + **GelSight Mini** 指尖触觉 + Intel RealSense D435 第三视角相机；人演示用 Meta Quest；遥操经 **OpenTeach**
- **入库日期：** 2026-07-08
- **一句话说明：** **策略无关** 的两阶段 **真机 RL** 管线：冻结 **视觉基策略**（人视频 Flow / ACT / DP / π₀.₅），用 **触觉残差策略** 在线修正接触丰富「最后一公里」；**无需离线触觉演示**；四任务 **40–80 min** 内把基策略 **5–40% → 85–100%** 成功率。

## 摘要级要点

- **问题：** 人视频/遥操等 **规模化视觉数据** 能训出强视觉策略，但 **接触力与局部几何** 相机测不到；触觉数据 **远小于视觉** 且难跨传感器泛化；端到端 **视触觉联合训练** 需大量配对演示。
- **核心范式：** 类比人类学习——**视觉模仿** 得任务级运动先验，**触觉在线练习** 学接触修正；把触觉适应写成 **冻结视觉策略上的残差校正**，而非从零训 visuo-tactile 策略。
- **Stage I（触觉感知 warm-start）：** 冻结基策略 **自主 rollout** 初始化 replay buffer；联合 **bootstrap flow-tactile critic** 与 **微调触觉编码器**（AnyTouch2 等）；**ControlTac** 轨迹级触觉增广扩接触样本。
- **Stage II（在线残差 RL）：** 轻量 **flow-tactile 残差 actor** 预测有界修正 $\mathbf{a}_t=\mathbf{a}_t^b+s_t\mathbf{a}_t^r$；**接口级策略无关**：共享 **物体关键点目标** + **基策略 action chunk** + 触觉，不读基策略内部隐状态。
- **奖励：** 归一化 **多感官塑形**——到达、抓取/接触、**生成物体流匹配**、触觉稳定 − 安全惩罚；终端成功 +1。
- **基策略族：** 人视频 **Flow Policy**（Im2Flow2Act / GenFlowRL / Dex4D 线，DINOv2+SAM+CoTracker3 关键点流）；遥操数据训 **ACT / DP / π₀.₅**。
- **四真机任务：** Peg-in-Hole（空间泛化+插入）、Charger Insertion（微小公差）、Cap Opening（工具+动态接触）、Box Opening（杠杆+小边缘对齐）。
- **主结果（Table 5）：** OmniTacTune 平均 **93.75%** vs PLD* **52.5%**、PLD Visual-Only **37.5%**、ViTAL **43.75%**；各任务终值 **100/100/90/85%**。
- **跨基策略（Peg-in-Hole，~50 min）：** 五类基策略均 **+40–60 pt** 提升至 **75–100%**；人视频 Flow 终值最高。
- **vs 更多演示：** 给 ACT+触觉 / RDP / π₀.₅+触觉 等 IL 基线 **额外 50 min 遥操数据**（50→90 条），仍落后 OmniTacTune **20–30 pt**——**在线试错** 优于堆触觉演示。
- **跨触觉表示：** AnyTouch2、Sparsh、T3 预训练编码器与 **低维 marker MLP** 均可适配；动态接触多的 Charger Insertion 上 T3/Sparsh 略逊于 AnyTouch2。
- **局限：** 继承真机 RL 的 **人工 reset** 与 **脆弱视触觉传感器磨损**；未来可接世界模型生成更逼真 visuo-tactile rollout。

## 核心论文摘录（MVP）

### 1) 触觉残差适应：视觉先验 + 触觉在线练习

- **链接：** <https://arxiv.org/html/2607.03723v1#S1>
- **摘录要点：** 视觉提供 **可扩展任务级行为**；触觉提供 **局部残差反馈** 完成物理交互最后一公里。离线视觉演示 **不含触觉监督**，故不宜 real-to-sim-to-real 或纯 DAgger 堆演示；转向 **真机 RL** 学残差。
- **对 wiki 的映射：**
  - [OmniTacTune（论文实体）](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md) — 「冻结视觉 + 触觉残差 RL」范式与 T-Rex 端到端触觉 VLA 的对照轴。

### 2) 两阶段真机 RL：warm-start critic/encoder + 在线残差

- **链接：** <https://arxiv.org/html/2607.03723v1#S3.SS3>
- **摘录要点：** Stage I 用基策略 rollout 填 buffer，**同时** 优化 flow-tactile critic $Q_\eta(q,z^f,z^\tau,a)$ 与触觉编码器（接触帧 + 重建正则）；**ControlTac** 在轨迹级合成 $\Delta F$ 触觉变体。Stage II 残差 actor 输入 $(q,z^f,g_t\cdot z^\tau,a^b,a^b_{t:t+K})$，$g_t$ 为接触门控；基策略 **全程冻结**。
- **对 wiki 的映射：**
  - [在 RL 中利用触觉反馈](../../wiki/queries/tactile-feedback-in-rl.md) — 真机残差 RL、密集多感官奖励与无仿真触觉路径。
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md) — 「残差适应」vs「端到端融合」第三路线。

### 3) 策略无关接口：生成流 + 基策略 chunk + 触觉

- **链接：** <https://arxiv.org/html/2607.03723v1#S3.SS2>、<https://arxiv.org/html/2607.03723v1#S3.SS4>
- **摘录要点：** Flow generator 从初始观测预测 **稀疏物体关键点子目标流**，供基策略与残差策略共享，并产生 **flow 匹配稠密奖励**；残差与奖励均 **不依赖** ACT/DP/π₀.₅ 内部表示，故同一残差学习器可挂接多架构。
- **对 wiki 的映射：**
  - [OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md) — 与 [Diffusion Policy](../../wiki/methods/diffusion-policy.md)、[VLA](../../wiki/methods/vla.md)、[Im2Flow2Act 系 Flow Policy](../../wiki/methods/imitation-learning.md) 的插件式关系。

### 4) 四任务评测与跨基策略/触觉表示泛化

- **链接：** <https://arxiv.org/html/2607.03723v1#S4>
- **摘录要点：** xArm7+GelSight Mini；基策略（人 Flow）初成功率 **40/10/5/5%**；训练 **50/40/60/80 min**（含 12 min warm-start）；对比 PLD* / PLD visual-only / ViTAL。跨 ACT/DP/π₀.₅ 与 AnyTouch2/markers/Sparsh/T3 均有效。
- **对 wiki 的映射：**
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)、[Manipulation 任务](../../wiki/tasks/manipulation.md) — 插装/工具使用/contact-rich 真机 benchmark 族。

### 5) 在线练习优于更多触觉演示

- **链接：** <https://arxiv.org/html/2607.03723v1#S4.SS3>
- **摘录要点：** ACT+触觉拼接、RDP、π₀.₅+触觉 token SFT 在 **+50 min 遥操** 后 Peg-in-Hole 仍 **60–65%**，OmniTacTune 达 **100%**（人 Flow 基座）。
- **对 wiki 的映射：**
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md) — 误区「堆演示 fine-tune 就够」的反例；与 [T-Rex](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)「需触觉 mid-training」形成 **数据预算–架构** 二维对照。

## 对 wiki 的映射（汇总）

- 沉淀实体页：[OmniTacTune 触觉残差适应（arXiv:2607.03723）](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md)
- 交叉补强：[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[触觉 RL Query](../../wiki/queries/tactile-feedback-in-rl.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md)、[T-Rex](../../wiki/entities/paper-trex-tactile-reactive-dexterous-manipulation.md)

## 当前提炼状态

- [x] 摘要、两阶段 RL、策略无关接口、ControlTac、四任务结果、跨基策略/触觉表示、vs IL 基线已摘录
- [x] 与项目页 [`sources/sites/omnitactune-project.md`](../sites/omnitactune-project.md) 互证

## BibTeX

```bibtex
@misc{yu2026omnitactune,
  title={OmniTacTune: Policy-Agnostic Real-World RL for Tactile Residual Adaptation of Visual Policies},
  author={Kelin Yu and Haode Zhang and Harish Ravichandar and Yunhai Han and Ruohan Gao},
  year={2026},
  eprint={2607.03723},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2607.03723}
}
```
