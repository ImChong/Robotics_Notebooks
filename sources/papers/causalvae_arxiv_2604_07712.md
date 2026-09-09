# CausalVAE as a Plug-in for World Models: Towards Reliable Counterfactual Dynamics（arXiv:2604.07712 / ECCV 2026）

> 来源归档（ingest）

- **标题：** CausalVAE as a Plug-in for World Models: Towards Reliable Counterfactual Dynamics
- **类型：** paper / latent world model / causal representation / counterfactual dynamics
- **arXiv：** <https://arxiv.org/abs/2604.07712>
- **会议：** ECCV 2026（README 引用）
- **代码：** <https://github.com/Dzyy123/CausalVAE-World-Models>（MIT；fork CausalMBRL / C-SWM）
- **作者：** Ziyi Ding、Xianxin Lai、Weiyu Chen、Xiao-Ping Zhang、Jiayu Chen（通讯）
- **机构：** 清华大学深圳国际研究生院；香港大学；INFIFORCE Intelligent Technology
- **入库日期：** 2026-09-09
- **一句话说明：** 在 latent WM 骨干上外挂 CausalVAE 因果层（DAG + 对齐弱监督），三阶段训练保事实预测、显著抬升干预/反事实检索（Physics 上 8 组基线 CF-H@1 平均 **+102.5%**）。

## 开源状态（仓库核查，2026-09-09）

- **已开源：** [`Dzyy123/CausalVAE-World-Models`](https://github.com/Dzyy123/CausalVAE-World-Models) 标注为官方实现；含 `cswm/models/causal_layer.py`、三阶段训练脚本与反事实评测入口。Checkpoints 与大媒体未随仓发布。

## 核心论文摘录（MVP）

### 1) Plug-in 因果结构分支

- **链接：** §3；Fig. 1
- **摘录要点：** 编码器得 object-centric \(z_t\)；CausalVAE 分支学 DAG 约束的因果潜变量 \(\tilde z_t\) 并解码回转移空间；可挂 AE/VAE/GNN/Modular/C-SWM 等骨干而不改其接口。
- **对 wiki 的映射：**
  - [paper-causalvae-world-models](../../wiki/entities/paper-causalvae-world-models.md)
  - [generative-world-models](../../wiki/methods/generative-world-models.md)

### 2) 三阶段训练

- **链接：** §3.4
- **摘录要点：** Stage 1 预训 encoder–transition；Stage 2 冻结骨干、训因果分支 + DAG；Stage 3 冻结 CausalVAE、alpha-gated fusion 精调转移。
- **对 wiki 的映射：** [paper-causalvae-world-models](../../wiki/entities/paper-causalvae-world-models.md) — 工程实践。

### 3) 干预式评测与 Physics 增益

- **链接：** §4；Tab. 1–2
- **摘录要点：** 四域 benchmark（Physics 3-body、2D Shapes、3D Cubes、Chemistry）；指标 H@1/MRR/CF-H@1/CF-MRR。Physics 上 GNN-NLL 代表设置 CF-H@1 **11.0→41.0**（+272.7%）；8 组配对基线 CF-H@1 平均 **+102.5%**。
- **对 wiki 的映射：**
  - [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md)
  - [paper-state-readout-decoupling](../../wiki/entities/paper-state-readout-decoupling.md) — 同组后续 latent WM 工作

## BibTeX

```bibtex
@inproceedings{ding2026causalvae,
  title     = {CausalVAE as a Plug-in for World Models: Towards Reliable Counterfactual Dynamics},
  author    = {Ding, Ziyi and Lai, Xianxin and Chen, Weiyu and Zhang, Xiao-Ping and Chen, Jiayu},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-causalvae-world-models.md`](../../wiki/entities/paper-causalvae-world-models.md)
- 代码归档：[`sources/repos/causalvae-world-models.md`](../repos/causalvae-world-models.md)
