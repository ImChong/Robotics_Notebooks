# Imagined Rollouts are Kinematic, Not Dynamic（arXiv:2607.05966）

> 来源归档（ingest）

- **标题：** Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure
- **类型：** paper / world-model diagnosis / kinematic-vs-dynamic / iKCE / DreamerV3
- **arXiv：** <https://arxiv.org/abs/2607.05966>（PDF：<https://arxiv.org/pdf/2607.05966.pdf>）
- **会场：** RSS Robot World Model Workshop 2026（workshop paper，9 页）
- **作者：** Finn Rasmus Schäfer、Korbinian Moller、Yuan Gao、Christian Oefinger、Sebastian Schmidt、Johannes Betz
- **机构：** 慕尼黑工业大学（TU Munich）— Autonomous Vehicle Systems Lab；Data Analytics and Machine Learning Group
- **入库日期：** 2026-07-27
- **一句话说明：** 把长程世界模型失败从笼统「误差累积」改写为 **运动学想象 vs 动力学想象**；提出 **imagined Kinematic-Consistency Error（iKCE）** 与摩擦扰动协议，在公开 DreamerV3 / DMC walker-walk 上给出「运动学非动力学」签名。

## 开源状态（核查，2026-07-27）

- **未开源：** arXiv 页与 PDF **无官方诊断代码仓**；实验基于 **已发布的 DreamerV3 checkpoint**（上游开源）与自建 iKCE / 扰动协议，本文 diagnost 实现本身 **未见独立 GitHub**。
- **复现边界：** 可按论文 Eq.(1) 与摩擦扫描复述协议；无可一键跑通的官方 diagnostic 包。

## 摘要级要点

- **中心论断：** 当前世界模型倾向按位置–速度–加速度外推（运动学），而非复现质量 / 摩擦 / 接触等约束（动力学）。
- **iKCE：** 对想象 rollout \(\{\hat{x}^{\mathrm{WM}}_t\}\) 相对闭式运动学零模型 \(\mathrm{kin}(\cdot)\) 的逐步 \(L_2\) 残差均值。
- **实例：** DreamerV3 @ DMC walker-walk；\(T{=}16\) 时 kinematic-null residual 约 **∼180×** 高于匹配真物理 rollout；摩擦扫描跨 gait-collapse 边界时 **策略回报崩塌而 iKCE 统计平坦**。
- **诊断窗口：** 需长于具身 **步态周期** 的 horizon 才能区分运动学 / 动力学想象。

## 核心论文摘录（MVP）

### 1) 运动学 vs 动力学 reframing

- **链接：** §I Central Claim；§II 与 Dreamer / MBPO 对照
- **摘录要点：** compounding error 正确但欠定；第三种失败叙事是 kinematic fallback。
- **对 wiki 的映射：**
  - [Imagined Rollouts…](../../wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md)
  - [运动学 vs 动力学可行](../../wiki/concepts/kinematic-vs-dynamic-feasibility.md)

### 2) iKCE 定义

- **链接：** §III.A Eq.(1)
- **摘录要点：** 复用 Gao 等训练期 kinematic-consistency loss 的形式，改为 **测试期** 诊断（前缀 i）。
- **对 wiki 的映射：**
  - [Imagined Rollouts…](../../wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md)
  - [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)

### 3) 扰动协议与双签名

- **链接：** §III–IV；friction sweep
- **摘录要点：** 签名是 **regime-invariance**（摩擦变、iKCE 不变）而非绝对幅度；平凡运动学预测器也会有高 iKCE。
- **对 wiki 的映射：**
  - [物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md) — 评测诊断族

## BibTeX

```bibtex
@inproceedings{schafer2026imagined,
  title     = {Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure},
  author    = {Sch\"afer, Finn Rasmus and Moller, Korbinian and Gao, Yuan and
               Oefinger, Christian and Schmidt, Sebastian and Betz, Johannes},
  booktitle = {RSS Robot World Model Workshop},
  year      = {2026},
  note      = {arXiv:2607.05966}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md`](../../wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md)
- 互链：[运动学 vs 动力学可行](../../wiki/concepts/kinematic-vs-dynamic-feasibility.md)、[DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)、[物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md)、[KineBench](../../wiki/entities/paper-kinebench.md)
- 策展入口：[微信 · 世界模型物理保真度](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
