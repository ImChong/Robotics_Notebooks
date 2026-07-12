# ContactMimic: Humanoid Object Interaction via Contact Control

> 来源归档（ingest）

- **标题：** ContactMimic: Humanoid Object Interaction via Contact Control
- **类型：** paper / humanoid / motion-tracking / contact-conditioning / loco-manipulation / hoi / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2607.08742>
- **提交日期：** 2026-07-09
- **项目页：** <https://lixinyao11.github.io/contactmimic-page/>（归档见 [`sources/sites/lixinyao11-contactmimic-github-io.md`](../sites/lixinyao11-contactmimic-github-io.md)）
- **机构：** University of Illinois Urbana-Champaign（UIUC）
- **作者：** Xinyao Li\*、Xialin He\*（共同一作）、Runpei Dong、Saurabh Gupta
- **硬件：** Unitree G1（29 DoF）；真机 5 条 HOI 动作 contact controllability 评测
- **仿真：** Isaac Lab + PhysX；PPO 训练 contact-conditioned keypoint tracker
- **入库日期：** 2026-07-12
- **一句话说明：** 在 keypoint tracking 之外显式跟踪 **per-body 二值接触指令**；用 **contact-following 奖励** 与 **打破关键点–接触相关性的轨迹增广**（label 翻转 / 去物体 / 膨胀碰撞几何）训练策略，使同一参考下可 **开启或抑制** 任务相关物理接触；HUMOTO 10 条仿真 + G1 真机 5 条验证，显著优于纯 keypoint 基线 BeyondMimic。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://lixinyao11.github.io/contactmimic-page/> | 方法概述视频、仿真/真机 contact ✔/✘ 对比 |
| 重定向 | [OmniRetarget（arXiv:2509.26633）](https://arxiv.org/abs/2509.26633) | 论文用于 HUMOTO→G1 重定向并提取接触标签 |
| 数据集 | [HUMOTO（ICCV 2025）](https://arxiv.org/abs/2505.10903) | 10 条人–物交互 clip 训练与评测 |
| keypoint 基线 | [BeyondMimic（arXiv:2508.08241）](https://arxiv.org/abs/2508.08241) | 纯 keypoint tracker 对照；接触指标显著落后 |
| 同类接触跟踪 | [SceneBot（arXiv:2606.27581）](https://arxiv.org/abs/2606.27581)、[ResMimic](https://arxiv.org/abs/2510.05070) | 均强调接触条件化；ContactMimic 突出 **运行时 contact 开关** 与 **数据增广解耦** |
| 同机构 HOI | [InterPrior（arXiv:2602.06035）](https://arxiv.org/abs/2602.06035) | UIUC 系物理 HOI 生成式控制姊妹线 |

## 摘要级要点

- **问题：** 擦白板、坐椅、推家具等任务的成功由 **哪一 body part 在何时接触哪一 object part** 定义，而非仅 keypoint 几何；纯 keypoint tracker 可摆出正确姿态却 **不产生有意义物理接触**。
- **接口：** $\pi_\theta(a_t \mid p_t, \bar{\mathbf{k}}_t, \bar{\mathbf{c}}_t)$；$\bar{\mathbf{c}}_t \in \{0,1\}^{|\mathcal{B}|}$ 覆盖骨盆、躯干、髋、膝、踝、肩、腕等可接触 link；**部署时可按 part 开关接触意图**（如坐椅但不靠椅背）。
- **奖励：** 标准 tracking + 正则 + **contact label matching** $r^{\mathrm{lm}}$（balanced accuracy 或 sparse 用 TP−λ·FP）+ **contact distance** $r^{\mathrm{cd}}$（应接触拉近、不应接触推远）。
- **数据增广（核心）：** 原始 HOI 数据中 keypoint 与 contact 强相关 → 策略可忽略 contact 指令。三种可组合增广：**① contact-label 翻转**；**② 移除物体并清零标签**；**③ 膨胀目标碰撞几何使重定向绕开物体**。消融：去掉增广后 contact controllability 显著下降。
- **数据管线：** HUMOTO clip → **OmniRetarget** → G1 参考 $\bar{\mathbf{q}}_{1:T}$；从 retarget 轨迹按 **1 cm 阈值** 提取 body–object-part 二值接触标签。
- **仿真：** 10 条多样 HOI（擦板、坐桌、靠椅背×2、脚踩椅、坐桌、靠桌、坐/蹲、踢椅、搬箱）；同一 keypoint 下 ✔/✘ contact 指令可切换接触数、冲量与关键关节力矩；**搬箱无需任务专用奖励**。
- **真机：** 5 条动作 wipe / sit-table / lean-backrest I&II / sit-squat；contact ✔ 与 ✘ 成功率多为 **5/5–10/10**。
- **vs BeyondMimic：** MPJPE 相当，但 contact bodies / impulse 与自由物体位移显著更高；kick chair、pick up box 等 **keypoint-only 几乎不移动物体**。
- **本体感知编码接触：** 线性探针在 policy 输入与 layer-2 表征上预测 runtime contact，F1 远高于 chance 与参考标签——**推理不需全身接触传感器**。
- **局限：** **每条动作单独训一条策略**（非 universal tracker）；依赖 **HUMOTO 高质量 HOI**；真机仅 G1、5 条动作。

## 核心摘录（面向 wiki 编译）

### 1) Contact-conditioned keypoint tracker 与奖励

- **策略：** 本体 $p_t$ + 参考 keypoint $\bar{\mathbf{k}}_t$ + 二值 contact map $\bar{\mathbf{c}}_t$ → PD 目标关节角。
- **$r^{\mathrm{lm}}$：** 比较实际 $c_{t,b,p}$ 与参考 $\bar{c}_{t,b,p}$；sparse-contact 动作用 TP−λ·FP 变体。
- **$r^{\mathrm{cd}}$：** 应接触对用高斯距离奖励；不应接触对在 $d<\delta$ 时惩罚。

### 2) 轨迹增广打破 keypoint–contact 相关

| 增广 | 行为 |
|------|------|
| Contact-label flipping | 保留轨迹，翻转任务相关 contact 标签 |
| Object removal | 移除物体、标签置零，keypoint 不变 |
| Inflated geometry | 膨胀碰撞 mesh，重定向绕开 → 远 keypoint + 零标签；可与 flipping 组合 |

- **论点：** 仅改 policy 输入与 reward **不足以**学到 genuine contact control；训练数据必须提供 **同 keypoint、异 contact** 配对。

### 3) 实验设置与 contact controllability 协议

- **平台：** G1 29 DoF，Isaac Lab；HUMOTO 10 clip（Table 1：擦板、椅、桌、箱等；kick/pick-up 为自由物体）。
- **四组测试轨迹：** near/far keypoints × contact ✔/✘；far 由 inflated geometry 产生。
- **指标：** contact bodies、contact impulse、关键关节力矩、MPJPE；自由物体另报 displacement。

### 4) 真机结果摘要（Table 2）

| Motion | contact ✔ | contact ✘ |
|--------|-----------|-----------|
| Wipe whiteboard | 5/5 | 5/5 |
| Sit in front of table | 4/5 | 5/5 |
| Lean on backrest I | 9/10 | 10/10 |
| Lean on backrest II | 10/10 | 9/10 |
| Sit and squat | 5/5 | 5/5 |

### 5) 与 SceneBot / ResMimic 的定位差异（策展）

- **SceneBot：** 单策略 + hindsight 场景重建 + per-link×{terrain,object} label，强调 **通才 tracker + 场景数据引擎**。
- **ResMimic：** GMT 先验 + 残差 + contact-tracking reward，**无运行时细粒度 contact 开关**。
- **ContactMimic：** **per-motion 策略** + **显式 body-level contact 指令** + **增广解耦**；强调 **同一 keypoint 下 contact on/off** 与 **无任务奖励的 loco-manipulation**（搬箱）。

## 对 wiki 的映射

- [ContactMimic](../../wiki/entities/paper-contactmimic.md) — 问题定义、方法归纳、实验与对照
- [人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md) — 「接触丰富场景 tracking」分支补充
- [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md) — contact-conditioned HOI tracking 扩展
- [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 物体交互 tracking 代表工作
- [InterPrior](../../wiki/entities/paper-interprior.md) — 同 UIUC HOI 研究线交叉引用

## 引用（项目页 BibTeX）

```bibtex
@article{li2026contactmimic,
  title   = {ContactMimic: Humanoid Object Interaction via Contact Control},
  author  = {Li, Xinyao and He, Xialin and Dong, Runpei and Gupta, Saurabh},
  journal = {arXiv preprint arXiv:2607.08742},
  year    = {2026}
}
```
