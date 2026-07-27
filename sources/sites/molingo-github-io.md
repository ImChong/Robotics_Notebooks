# MoLingo 项目页（hynann.github.io/molingo）

- **标题：** MoLingo: Motion-Language Alignment for Text-to-Motion Generation
- **类型：** site / project-page
- **URL：** <https://hynann.github.io/molingo/MoLingo.html>
- **会议：** CVPR 2026
- **arXiv：** <https://arxiv.org/abs/2512.13840> — 归档见 [`sources/papers/molingo_arxiv_2512_13840.md`](../papers/molingo_arxiv_2512_13840.md)
- **代码：** <https://github.com/hynann/MoLingo> — 归档见 [`sources/repos/molingo.md`](../repos/molingo.md)
- **作者单位：** University of Tübingen / Tübingen AI Center / Max Planck Institute for Informatics / Imperial College London / Zuse School ELIZA
- **入库日期：** 2026-07-27

## 一句话摘要

图宾根–MPI 等团队的 **文本→人体运动（T2M）** 官方项目页：用 **语义对齐连续潜空间（SAE）** + **掩码自回归 rectified flow** + **多 token 交叉注意力文本条件**，在 HumanML3D 上刷 FID / R-Precision，并演示 **Unitree G1 + PHC 策略** 跟踪生成动作。

## 项目页核查（步骤 2.5 · 2026-07-27）

| 核查项 | 结论 |
|--------|------|
| **Code / Resources** | 项目页正文写 “We will release our code and models”；**实际仓库已开放**：GitHub [`hynann/MoLingo`](https://github.com/hynann/MoLingo)（Apache-2.0），README 链回本项目页与 arXiv |
| **开放程度** | **已开源（部分下游待发布）**：评测、生成 demo、SAE / MoLingo 训练脚本与预训练权重下载脚本已发布；README **TODO** 仍勾选 **G1 tracking pipeline** 未发布 |
| **数据集** | 依赖公开 **HumanML3D（263D）** 与 **HumanML3D-272**（MotionStreamer 处理）；SAE 训练另需作者提供的 HumanML3D–BABEL frame-level 特征包（uni-tuebingen Nextcloud 链） |
| **模型 checkpoint** | `prepare/download_models.sh` 可下载预训练 SAE / 生成模型（含 272D 更新说明，2026-03-07） |

- **代码：** <https://github.com/hynann/MoLingo>
- **数据集：** HumanML3D / HumanML3D-272（第三方管线）+ BABEL 帧级标注交集特征包
- **模型 checkpoint：** 已通过官方 `prepare/download_models.sh` 发布

## 公开信息要点（项目页归纳）

### Abstract 主张

- 在 **连续运动潜空间** 上做去噪生成，研究两件事：**(1)** 如何构建 **语义对齐**、更利于扩散的潜空间；**(2)** 如何注入文本条件使动作更贴描述。
- **SAE**：用 **帧级文本标签** 训练语义对齐运动编码器，让相近语义的 latent 靠近。
- **文本条件**：对比 **单 token** vs **多 token cross-attention**，后者在真实感与 text–motion 对齐上更好。
- 组合 **语义对齐 latent + 自回归生成 + cross-attention** → 标准指标与用户研究上的 SOTA 宣称。

### Method Overview

| 侧 | 内容 |
|----|------|
| Left · SAE | 运动序列 encoder–decoder + 并行文本支路；帧级标签 → class token；**cosine similarity** 对齐 motion latent |
| Right · AR flow | Transformer decoder 产出条件向量 \(z\)，引导 MLP 迭代精炼 latent；训练时随机 mask 替换为可学习 token；推理从全 mask 迭代去噪再 decode |

### 对比与消融（项目页视频/案例）

- **对比基线：** MARDM、ACMDM、MotionStreamer（同 HumanML3D 测试提示）
- **表示说明：** ACMDM/MARDM 侧用 SimpLify 可视化；MotionStreamer 用 **272D**；本文主表示 **263D**（社区常用）+ SimpLify；G1 对照实验用 **272D** 便于重定向
- **SAE vs VAE：** 同真实感下，SAE 更忠实执行多阶段/复合指令（放下箱子再跑、俯卧撑再起身、边走边踢等）

### Unitree G1 跟踪演示

- 将 MotionStreamer / MoLingo 生成动作 **重定向到 G1**，接 **预训练 RL tracking（PHC 策略）** 评估物理可跟踪性
- 案例：挥网球拍、芭蕾、向后运球等；宣称 MoLingo 足地接触更稳、更少控制器纠偏

### BibTeX

```bibtex
@inproceedings{he2026molingo,
  title={MoLingo: Motion–Language Alignment for Text-to-Human Motion Generation},
  author={He, Yannan and Tiwari, Garvita and Zhang, Xiaohan and Bora, Pankaj and Birdal, Tolga and Lenssen, Jan Eric and Pons-Moll, Gerard},
  booktitle={CVPR},
  year={2026}
}
```

## 对 wiki 的映射

- 论文实体：[paper-molingo](../../wiki/entities/paper-molingo.md)
- 交叉：[`diffusion-motion-generation`](../../wiki/methods/diffusion-motion-generation.md)、[`hy-motion-1`](../../wiki/methods/hy-motion-1.md)、[`dart-control`](../../wiki/methods/dart-control.md)、[`awesome-text-to-motion-zilize`](../../wiki/entities/awesome-text-to-motion-zilize.md)、[`phc`](../../wiki/entities/phc.md)、[`paper-notebook-humanml3d`](../../wiki/entities/paper-notebook-humanml3d.md)、[`paper-phygile`](../../wiki/entities/paper-phygile.md)
