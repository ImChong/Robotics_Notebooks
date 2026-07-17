# FMPose3D: monocular 3D pose estimation via flow matching（arXiv:2602.05755）

> 来源归档（ingest）

- **标题：** FMPose3D: monocular 3D pose estimation via flow matching
- **简称：** FMPose3D
- **类型：** paper / 3D pose estimation / flow matching / generative lifting / CVPR 2026
- **venue：** CVPR 2026
- **原始链接：**
  - arXiv abs：<https://arxiv.org/abs/2602.05755>
  - arXiv PDF：<https://arxiv.org/pdf/2602.05755>
  - 项目页：<https://xiu-cs.github.io/FMPose3D/>
  - 代码：<https://github.com/AdaptiveMotorControlLab/FMPose3D>
- **机构：** 洛桑联邦理工学院（EPFL），Adaptive Motor Control Lab（Mackenzie Weygandt Mathis）
- **作者：** Ti Wang、Xiaohang Yu、Mackenzie Weygandt Mathis
- **入库日期：** 2026-07-17
- **一句话说明：** 将单目 **2D→3D 姿态提升** 表述为 **条件 Flow Matching 分布传输**：用 ODE 速度场在少量积分步内从 Gaussian 噪声生成多样 3D 假设，再以 **RPEA**（重投影后验期望聚合）融合为单帧预测；在 Human3.6M / MPI-INF-3DHP 与 Animal3D / CtrlAni3D 上达到 SOTA，推理显著快于扩散基线 DiffPose。

## 摘要级要点

- **问题：** 单目 3D 姿态 **深度歧义 + 遮挡** 使确定性回归易塌缩到均值；扩散式多假设方法准确但 **逐步去噪步数多、推理慢**。
- **FMPose3D：** 条件 **CFM** 学习 $v_\theta(x_t,t,c)$，$c=x^{2D}$；训练用线性插值路径 $x_t=(1-t)x_0+t x_1$ 与目标速度 $v_t=x_1-x_0$；推理 **Euler 积分** $S$ 步（默认 **$S=3$**）从噪声到 3D 关键点。
- **多假设：** 不同噪声种子 → 多样 3D 假设；**FHA** 将原图与水平翻转视为两组假设一并送入 RPEA。
- **RPEA：** 以 2D 重投影误差作伪似然，对 Top-K 候选做加权期望，逼近 MSE 下 **Bayes 最优 MMSE**；支持 joint-wise / pose-wise。
- **骨干：** **并行 GCN（局部骨架）+ Self-Attention（全局）** 融合预测速度场（Human3.6M 上 MPJPE **49.3 mm**）。
- **开源：** 官方实现 **Apache-2.0**；预训练权重 **Hugging Face** `MLAdaptiveIntelligence/FMPose3D`（模型 **CC BY-NC-ND**）；**PyPI** `fmpose3d`；2026-03 集成进 **DeepLabCut** 动物管线。

## 核心摘录（面向 wiki 编译）

### 1) 2D→3D lifting + Flow Matching 公式

- **链接：** 论文 §3.1、Fig. 1–2；[项目页 Overview](https://xiu-cs.github.io/FMPose3D/)
- **摘录要点：**
  - 输入 2D 关节 $x^{2D}\in\mathbb{R}^{J\times 2}$，输出 3D $x^{3D}\in\mathbb{R}^{J\times 3}$。
  - 损失 $\mathcal{L}_{\text{CFM}}=\|v_\theta(x_t,t,c)-(x_1-x_0)\|_2^2$。
  - 推理：$\hat{x}^{3D}=x_0+\int_0^1 v_\theta(x_t,t,c)\,dt$，离散为 $S$ 步 Euler。
- **对 wiki 的映射：**
  - [FMPose3D](../../wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md) — 训练/推理 Mermaid 与 CFM 机制

### 2) RPEA 多假设聚合

- **链接：** 论文 §3.2、Fig. 2–3
- **摘录要点：**
  - 采样 $N$ 个假设；joint-wise：每关节按重投影损失筛 Top-K，再 softmax 权重聚合。
  - Human3.6M：baseline **49.3 mm** → FHA + RPEA（$N=40$）**47.3 mm**，优于 DiffPose **49.7 mm**。
  - 相对 Mean / JPMA，RPEA 更能利用增加假设数，且 joint-wise 版保持解剖一致性（P-MPJPE 与 Mean 相当，pose-wise RPEA P-MPJPE 最优）。
- **对 wiki 的映射：**
  - [FMPose3D](../../wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md) — RPEA 与 FHA 小节

### 3) 人与动物基准 + 速度

- **链接：** 论文 §4、Table 1–5
- **摘录要点：**
  - **Human3.6M**（CPN 2D）：MPJPE **47.3**（RPEA）；**MPI-INF-3DHP** 跨数据集零微调 PCK/AUC 最佳。
  - **Animal3D** P-MPJPE **61.5** vs AniMer **80.4**；**CtrlAni3D** **44.0** vs AniMer **44.1**（联合训练，动物评测未用 RPEA）。
  - **速度（RTX 4090）：** FMPose3D $S=3,N=1$ → **160.11 FPS**；$N=40$ → **145.59 FPS**；DiffPose 50 步 **3.36 FPS**，DDIM-5 步 **27.15 FPS**（$N=40$ 时 FMPose3D 约 **5.4×** 快于 DiffPose）。
- **对 wiki 的映射：**
  - [FMPose3D](../../wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md) — 实验与推理速度表

### 4) 工程发布与 DeepLabCut 集成

- **链接：** [GitHub README](https://github.com/AdaptiveMotorControlLab/FMPose3D)、[PyPI fmpose3d](https://pypi.org/project/fmpose3d/)
- **摘录要点：**
  - `pip install fmpose3d`；动物可选 `fmpose3d[animals]`（DeepLabCut 依赖）。
  - 权重首次推理自动从 Hugging Face 下载；提供 **in-the-wild 单图 demo** 与 **Inference API**。
  - 动物 demo 自动下载 SuperAnimal-Quadruped 2D + FMPose3D 3D lifter 双 checkpoint。
  - README News：**2026-03 集成 DeepLabCut**。
- **对 wiki 的映射：**
  - [sources/repos/fmpose3d.md](../repos/fmpose3d.md) — 仓库与权重侧归档
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md) — 动物/关键点上游选项

## 对 wiki 的映射（汇总）

- [paper-fmpose3d-monocular-3d-pose-flow-matching.md](../../wiki/entities/paper-fmpose3d-monocular-3d-pose-flow-matching.md) — 主沉淀页
- 交叉更新：[motion-retargeting-pipeline.md](../../wiki/concepts/motion-retargeting-pipeline.md)

## 引用（项目页 BibTeX）

```bibtex
@misc{wang2026fmpose3dmonocular3dpose,
      title={FMPose3D: monocular 3D pose estimation via flow matching},
      author={Ti Wang and Xiaohang Yu and Mackenzie Weygandt Mathis},
      year={2026},
      eprint={2602.05755},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.05755},
}
```
