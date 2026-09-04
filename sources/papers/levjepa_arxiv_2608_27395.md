# LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics（arXiv:2608.27395）

> 来源归档（ingest）

- **标题：** LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics
- **类型：** paper / JEPA / self-supervised video / video representation / collapse-free SSL
- **arXiv：** <https://arxiv.org/abs/2608.27395>（PDF：<https://arxiv.org/pdf/2608.27395.pdf>）
- **项目页：** <https://levjepa.github.io/>
- **代码：** <https://github.com/MLO-lab/LeVJEPA>（MIT；`module.py` 改编自 Meta V-JEPA，**CC BY-NC 4.0**）
- **权重：** Hugging Face [`galilai-group/LeVJEPA-VideoMix-Large`](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large)（ViT-L/16 · **CC BY-NC 4.0**）
- **作者：** Lukas Kuhn、Lucas Maes、Giuseppe Serra、Quentin Le Lidec、Yann LeCun、Randall Balestriero\*、Florian Buettner\*（\* equal advising）
- **机构：** 德国癌症研究中心（DKFZ）/ 德国癌症联盟（DKTK）；法兰克福歌德大学；蒙特利尔学习算法研究所（Mila）/ 蒙特利尔大学；布朗大学；纽约大学 Courant；先进机器智能实验室（AMI Labs）
- **入库日期：** 2026-09-04
- **一句话说明：** 把 [LeJEPA](https://arxiv.org/abs/2511.08544) 的 **不变性损失 + SIGReg** 接到视频 Transformer：不要 EMA 目标编码器、predictor、stop-gradient 或像素重建；均匀随机丢掉 95% token 反而涨点，并免费得到 **block-causal** 逐帧表征。

## 开源状态（项目页 + 仓库核查，2026-09-04）

- **已开源：** 项目页互链 [MLO-lab/LeVJEPA](https://github.com/MLO-lab/LeVJEPA)（166★，2026-08-28 建仓）。完整预训练配方（Hydra + Lightning + Walking Tours Lance 数据管线）+ HF **LeVJEPA-VideoMix-Large**（303M，VideoMix 1.8M clips）。
- **许可证分层：** 仓库主体 **MIT**；`module.py` 与发布权重 **CC BY-NC 4.0**（改编自 facebookresearch/jepa）。商用复现需避开该文件 / 权重或自行替换骨干。
- **默认可复现路径：** README 默认在公开 [Walking Tours](https://huggingface.co/datasets/shawshankvkt/Walking_Tours) 上训 ViT-B/16（需自备 YouTube 下载 ~25 GB + Lance 编码 ~33 GB）；论文主表的 K710 20% 对照需自备 Kinetics。

## 摘要级要点

- **目标：** \(\mathcal L=\mathcal L_{\mathrm{inv}}+\lambda\,\mathcal L_{\mathrm{SIGReg}}\)，\(\lambda=0.02\) 全实验不调；可训练件只剩 encoder + 小 projector。
- **视图：** 16 帧 clip；1 个全局 \(224^2\) + \(V\) 个局部 \(96^2\)（默认 \(V=4\)，可到 10）；共享同一时间窗。
- **Token dropping：** \(\rho=0.95\) 均匀随机丢 patch；IN1K 从 33.9%（\(\rho=0\)）单调升到 47.6%；tube masking 反而掉到 39.6%。
- **Attention：** block-causal（帧内双向、跨帧因果）51.2% vs 双向 50.7%；推理可按帧增量编码，不必重编历史。
- **对照（同数据重训）：** epoch-matched 对 V-JEPA 2 省 **5.6–20.8×** FLOPs；FLOP-matched ViT-B：IN1K **61.0** / SSv2 40.4 / K400 44.6，相对最强视频基线 IN1K **+7.6**。
- **缩放：** ViT-L/16 @ VideoMix 100 epoch → 冻结 attentive probe IN1K **69.5%**、SSv2 **55.0%**（与 HF 卡一致；项目页摘要曾写 67.5%，以论文 §5.4 / 权重卡为准）。

## 核心论文摘录（MVP）

### 1) 坍塌对策从「架构不对称」换成 SIGReg

- **链接：** §3；Eq. (1)–(3)；Fig. 1
- **摘录要点：** 全局–局部 MSE 两端都反传，单独最小化会塌成常数；SIGReg 用 Cramér–Wold + Epps–Pulley 把 embedding 约束到各向同性高斯，从分布上排除坍塌。无需 EMA teacher / predictor / stop-gradient。
- **对 wiki 的映射：**
  - [LeVJEPA](../../wiki/entities/paper-levjepa.md)
  - [WCM](../../wiki/entities/paper-wcm-world-critic-model.md) — 同属 LeJEPA + SIGReg 家族，用在 VLA critic 而非视频预训练。

### 2) Token dropping 是增强不是近似

- **链接：** §4.1；Fig. 4
- **摘录要点：** 丢掉越多 IN1K 越高；tube 模式永久挡住大部分场景，均匀随机才构成可识别的时空样本。SSv2 在短日程上高 \(\rho\) 会掉，加长训练可在同等算力下挽回。
- **对 wiki 的映射：**
  - [V-JEPA 2](../../wiki/entities/paper-vjepa2.md) — 对照：掩码是预测任务的一部分，不是「唯一观测」。

### 3) Block-causal 免费、密集 patch 结构涌现

- **链接：** §4.4；§5.5；Fig. 3
- **摘录要点：** 因果注意力不掉点；只监督 `[cls]`，patch PCA 仍能把物体与背景分开——V-JEPA 2 无此结构，V-JEPA 2.1 靠显式 dense loss 才有。
- **对 wiki 的映射：**
  - [世界模型物理保真输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md) — 因果逐帧表征是后续 latent 规划 / 流式 WM 的前置，本文本身不做动作条件 MPC。

## BibTeX

```bibtex
@misc{kuhn2026levjepaefficientscalable,
      title={LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics},
      author={Kuhn, Lukas and Maes, Lucas and Serra, Giuseppe and Le Lidec, Quentin and LeCun, Yann and Balestriero, Randall and Buettner, Florian},
      year={2026},
      eprint={2608.27395},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.27395},
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-levjepa.md`](../../wiki/entities/paper-levjepa.md)
- 代码：[`sources/repos/levjepa.md`](../repos/levjepa.md)
- 项目页：[`sources/sites/levjepa-github-io.md`](../sites/levjepa-github-io.md)
