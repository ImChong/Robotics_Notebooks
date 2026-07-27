# DWM: Separating World Effects from Actions in Latent World Models（arXiv:2607.18715）

> 来源归档（ingest）

- **标题：** DWM: Separating World Effects from Actions in Latent World Models
- **短名：** DWM（Decomposed World Model / Separating World Effects）
- **类型：** paper / latent world model / action-invariant dynamics / CEM planning
- **arXiv：** <https://arxiv.org/abs/2607.18715>（PDF：<https://arxiv.org/pdf/2607.18715.pdf>）
- **项目页 / 代码：** **未发现**（截至 **2026-07-27** 论文页与检索均无官方项目页或 GitHub）
- **作者：** Yi-Ge Zhang、Tianqi Du、Qi Zhang、Yisen Wang（Corresponding：Yisen Wang）
- **机构：** 北京大学（PKU）
- **入库日期：** 2026-07-27
- **一句话说明：** 在 latent WM 训练目标层把转移拆成 **动作无关世界效应** 与 **动作诱导残差**：辅助 world head + 对比不变约束 + 与 pred head 正交；**不改推理架构**；在 PushT-W / Reacher-W / TwoRoom-W 上 CEM 规划成功率平均 **+13.1 pp**。

## 命名消歧（CRITICAL）

- **本文 DWM ≠** 本库已有 [Dexterous World Models（DWM，arXiv:2512.17907）](../../wiki/methods/dwm.md)。
- 入库实体页标题固定为 **「DWM（Separating World Effects）」**；与 dexterous DWM 仅做消歧互链，勿合并节点。

## 开源状态（核查，2026-07-27）

- **未开源：** 无项目页、无官方代码仓、无公开 checkpoint；论文未给出「code will be released」URL。wiki 须写明 **未开源**，源码运行时序图 **不适用**。

## 摘要级要点

- **诊断：** 单目标 next-latent 把 agent 效应与环境自主动态缠在一起；平坦基准掩盖问题，重力/漂移 W-variant 上单头 LeWM 失败。
- **方法：** 保留 pred head（推理唯一）；训练加 world head（动作扰动下预测稳定 + 状态可分）与正交约束，诱导 \(\Delta z=\Delta z^{\mathrm{world}}+\Delta z^{\mathrm{action}}\)。
- **基准：** PushT-W（重力滑移）、Reacher-W、TwoRoom-W；对照 flat 原版；另 Ball-in-Cup **+6.0%**。
- **规划：** latent 空间 CEM；推理路径与基座相同。

## 核心论文摘录（MVP）

### 1) World / action 分解定义

- **链接：** §3.2；Eq. (4)–(5)
- **摘录要点：** \(\Delta z^{\mathrm{world}}\) 为当前动作置零时的期望转移；action 为残差。
- **对 wiki 的映射：**
  - [DWM Separating](../../wiki/entities/paper-dwm-separating-world-effects.md)
  - [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md) — **动作 vs 世界效应分解** 族。

### 2) 监督目标与 CEM 增益

- **链接：** §4；§1 贡献；W-variant 平均 +13.1%
- **摘录要点：** world-contrastive + orthogonality；CEM success 提升 12.0 / 10.7 / 16.7 pp。
- **对 wiki 的映射：**
  - [V-JEPA 2](../../wiki/entities/paper-vjepa2.md) — latent 规划对照。
  - [methods/dwm.md](../../wiki/methods/dwm.md) — **仅消歧**。

### 3) 开源边界

- **链接：** 全文无 Code Availability URL
- **摘录要点：** 截至入库日 **未开源**。
- **对 wiki 的映射：** 实体页「工程实践 / 源码运行时序图」。

## BibTeX

```bibtex
@article{zhang2026dwm,
  title   = {DWM: Separating World Effects from Actions in Latent World Models},
  author  = {Zhang, Yi-Ge and Du, Tianqi and Zhang, Qi and Wang, Yisen},
  journal = {arXiv preprint arXiv:2607.18715},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-dwm-separating-world-effects.md`](../../wiki/entities/paper-dwm-separating-world-effects.md)
- 消歧对照：[`wiki/methods/dwm.md`](../../wiki/methods/dwm.md)（Dexterous World Models）
- 策展语境：[`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
