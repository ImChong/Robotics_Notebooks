# PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics（arXiv:2607.20653）

> 来源归档（ingest）

- **标题：** PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics
- **类型：** paper / deformable dynamics / world model / differentiable MPM / residual correction
- **arXiv：** <https://arxiv.org/abs/2607.20653>（PDF：<https://arxiv.org/pdf/2607.20653.pdf>）
- **作者：** Haocheng Yin\*、Shuohan Tao\*、Yongsheng Chen、Lu Gan（\* equal contribution）
- **机构：** 佐治亚理工学院（Georgia Institute of Technology）
- **入库日期：** 2026-07-27
- **一句话说明：** 可微 **MLS-MPM** 动力学骨干 + **Material from Motion（MfM）** 从视觉推断粒子材料参数与置信度 + **Residual from Dynamics（RfD）** 在网格更新与 G2P 之间修正接触/摩擦/模型误差；真机可变形序列上相对 PhysTwin 等基线降低 Chamfer / tracking loss，并支持置信度引导探索。

## 开源状态（核查，2026-07-27）

- **未开源：** arXiv abs / HTML 正文 **无官方 GitHub / 项目页 / 权重链接**；GitHub 检索到的同名仓库与本文无关。论文局限节写明当前实现聚焦弹性 / 弹塑性，未宣称公开代码。
- **复现边界：** 可复述方法（MfM → MPM → RfD 顺序训练）与所报指标，但 **无可运行官方实现**。

## 摘要级要点

- **瓶颈：** 逐物体优化材料慢且不泛化；端到端学习外推差且易破坏物理结构。
- **骨架：** 可微 **MLS-MPM + APIC**；Fixed Corotated Elasticity / von Mises Plasticity 分支由 MfM 的 \(\pi\) 选择。
- **MfM：** 短窗观测运动 → Graph U-Net → 每粒子 \(\phi_p=(\log E_p,\nu_p)\) + 置信度 \(c_p\)；仿真预训练后可在线适应新物体。
- **RfD：** 在 grid update 与 G2P 之间对节点速度加有界 \(\Delta\mathbf{v}\)（FiLM 稀疏 3D U-Net，输出层零初始化）。
- **数据 / 评测：** 12 段真机人手可变形操纵（绳 / 毛巾 / 毛绒 / 橡皮泥）；相对 PhysTwin：弹性 CD **−43.7%**、弹塑性 **−30.5%**；RfD 相对 MfM-only 再降 CD **13.1% / 17.8%**。

## 核心论文摘录（MVP）

### 1) 物理骨架 + 双前馈

- **链接：** §3–4.2；MLS-MPM P2G / grid / G2P
- **摘录要点：** 保留可微 MPM 的粒子–网格结构与本构；MfM 替换逐物体标定；RfD 吸收解析模型无法表达的接触 / 摩擦 / 系统偏差。
- **对 wiki 的映射：**
  - [PhysCoRe](../../wiki/entities/paper-physcore.md)
  - [物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md) — 「物理混合」族

### 2) MfM 材料从运动推断

- **链接：** §4.2.1；置信度与主动探索
- **摘录要点：** Fourier 特征编码轨迹 → Graph U-Net；置信度与实际变形区域对齐；KUKA 探测实验中置信度仅在被主动变形区域上升。
- **对 wiki 的映射：**
  - [PhysCoRe](../../wiki/entities/paper-physcore.md) — 在线材料识别

### 3) RfD 残差动力学修正

- **链接：** §4.2.2；Table 4
- **摘录要点：** 有界残差保持物理结构；弹塑性物体收益更大，印证「材料归 MfM、残差归 RfD」。
- **对 wiki 的映射：**
  - [PhysCoRe](../../wiki/entities/paper-physcore.md)
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 对照纯像素 WM

## BibTeX

```bibtex
@article{yin2026physcore,
  title   = {PhysCoRe: Physics-Corrected Residual World Models for Material-Aware Deformable Dynamics},
  author  = {Yin, Haocheng and Tao, Shuohan and Chen, Yongsheng and Gan, Lu},
  journal = {arXiv preprint arXiv:2607.20653},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-physcore.md`](../../wiki/entities/paper-physcore.md)
- 互链：[物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[运动学 vs 动力学可行](../../wiki/concepts/kinematic-vs-dynamic-feasibility.md)、[VT-WAM](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md)
- 策展入口：[微信 · 世界模型物理保真度](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
