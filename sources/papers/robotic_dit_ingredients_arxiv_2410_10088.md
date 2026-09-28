# The Ingredients for Robotic Diffusion Transformers（arXiv:2410.10088）

> 论文来源归档（ingest）

- **标题：** The Ingredients for Robotic Diffusion Transformers
- **作者：** Sudeep Dasari, Oier Mees, Sebastian Zhao, Mohan Kumar Srirama, Sergey Levine（CMU / UC Berkeley 等）
- **类型：** paper / diffusion-policy / transformer / manipulation / bimanual
- **arXiv：** <https://arxiv.org/abs/2410.10088> · PDF：<https://arxiv.org/pdf/2410.10088.pdf>
- **项目页：** <https://dit-policy.github.io/> — [`sources/sites/dit-policy-github-io.md`](../sites/dit-policy-github-io.md)
- **代码：** <https://github.com/SudeepDasari/dit-policy> — [`sources/repos/sudeepdasari_dit_policy.md`](../repos/sudeepdasari_dit_policy.md)
- **数据：** [BiPlay](https://huggingface.co/datasets/oier-mees/BiPlay)（7023 clips，~10h 双臂 ALOHA）
- **入库日期：** 2026-09-28
- **一句话说明：** 系统研究 **扩散 Transformer 策略** 的设计要素：**ResNet 分相机 tokenizer + FiLM 语言 + adaLN-Zero 解码器** 组成 **DiT-Block Policy**；长时域 ALOHA / DROID Franka SOTA；相对 [Diffusion Policy](../../wiki/methods/diffusion-policy.md) 的 naive cross-attn Transformer 更易训。

## 核心摘录（面向 wiki 编译）

### 1) 为何 U-Net DP 之外需要「配方」

- **要点：** 原文指出 DP 论文中 cross-attention Transformer 噪声网络 **极难调参**；社区多沿用 U-Net，但 U-Net 对动作平滑等假设限制场景；高容量 Transformer + 扩散应可缩放但缺稳定架构。
- **对 wiki 的映射：** [`wiki/entities/paper-diffusion-policy.md`](../../wiki/entities/paper-diffusion-policy.md)、[`roadmap/depth-robotics-diffusion-dit-flow.md`](../../roadmap/depth-robotics-diffusion-dit-flow.md)

### 2) adaLN-Zero 稳定扩散 Transformer 策略

- **要点：** 用 **adaptive LayerNorm（adaLN-Zero）** 替代标准 cross-attention 解码块，将观测 encoder embedding 与扩散步 \(k\) 注入 LayerNorm scale/shift；初始化 output projection 为 0（DiT 图像生成同族 trick）；长时域（1500+ 步）任务成功率 **+30%** 量级。
- **对 wiki 的映射：** [`wiki/entities/paper-dit-scalable-diffusion-transformers.md`](../../wiki/entities/paper-dit-scalable-diffusion-transformers.md)

### 3) 观测 token 化：分相机 ResNet-26 + 本体 dropout

- **要点：** 各相机 **独立 ResNet-26**（非单共享 encoder）；DistilBERT 文本经 **FiLM** 调制视觉层；本体维 **observation dropout** 防捷径；encoder 用 Octo 式 block attention；相对 ViT-only 等 **+40%** 量级。
- **对 wiki 的映射：** [`wiki/methods/action-chunking.md`](../../wiki/methods/action-chunking.md)

### 4) Action chunk 与推理

- **要点：** 训练预测 **H=100** action chunk；DDPM 训练 \(k=100\) 步、cosine schedule；推理 deterministic sampling **10 步**；配合 temporal ensembling。
- **对 wiki 的映射：** [`wiki/concepts/receding-horizon-policy-execution.md`](../../wiki/concepts/receding-horizon-policy-execution.md)

## 开源状态（步骤 2.5，2026-09-28）

| 资源 | 状态 |
|------|------|
| [SudeepDasari/dit-policy](https://github.com/SudeepDasari/dit-policy) | **已开源**（MIT；`finetune.py` + `agent=diffusion` DiT-Block；U-Net 等 baseline） |
| [BiPlay](https://huggingface.co/datasets/oier-mees/BiPlay) | **已发布**（HF Dataset） |
| 预训练视觉表征 | `download_features.sh` / CMU data4robotics release |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 升格实体：[`wiki/entities/paper-robotic-dit-ingredients-dit-block-policy.md`](../../wiki/entities/paper-robotic-dit-ingredients-dit-block-policy.md)
