---
type: entity
tags:
  - paper
  - dit
  - diffusion
  - transformer
  - generative-model
  - latent-diffusion
  - imagenet
  - iccv2023
  - berkeley
  - nyu
  - meta
status: complete
updated: 2026-11-07
arxiv: "2212.09748"
code: https://github.com/facebookresearch/DiT
related:
  - ../concepts/diffusion-transformer.md
  - ../concepts/diffusion-model.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
  - ../methods/generative-world-models.md
  - ./paper-dit4dit-video-action-model.md
  - ../overview/ai-architecture-map.md
  - ../overview/rcl-awesome-wam-technology-map.md
original:
  - ../../sources/papers/peebles_dit_arxiv_2212_09748.md
sources:
  - ../../sources/papers/peebles_dit_arxiv_2212_09748.md
  - ../../sources/sites/dit-wpeebles-com.md
  - ../../sources/repos/facebookresearch-dit.md
summary: "DiT（ICCV 2023，arXiv:2212.09748，Berkeley/NYU/Meta）：LDM 中用 patch ViT + AdaLN 替换 U-Net；Gflops 与 FID 强相关；ImageNet 256 FID 2.27；facebookresearch/DiT 已开源。"
---

# DiT：可扩展的 Transformer 扩散模型

**DiT**（*Scalable Diffusion Models with Transformers*，[arXiv:2212.09748](https://arxiv.org/abs/2212.09748)，[项目页](https://www.wpeebles.com/DiT)，[代码](https://github.com/facebookresearch/DiT)）由 **William Peebles**（加州大学伯克利分校 UC Berkeley）、**Saining Xie**（纽约大学 NYU）提出（Peebles 实习于 **Meta AI FAIR**）：在 **隐空间扩散（LDM）** 中用 **patch 化 Vision Transformer** 作去噪骨干，系统验证 **前向 Gflops 与生成质量（FID）** 的可预测缩放，并在 class-conditional ImageNet 256×256 / 512×512 上取得发表时扩散模型中最好的 FID。

## 一句话定义

**扩散去噪不必绑 U-Net：把带噪 latent 当 patch token 序列，用 AdaLN 注入时间步与类别，算力（Gflops）比参数量更能决定 FID。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DiT | Diffusion Transformer | 本文：Transformer 扩散去噪骨干 |
| LDM | Latent Diffusion Model | 在 VAE latent 上扩散，降算力 |
| AdaLN | Adaptive Layer Normalization | 用 \(t\)/类别调制 LayerNorm，本文最佳条件注入 |
| CFG | Classifier-Free Guidance | 训练随机丢类标签，推理混合条件/无条件噪声 |
| FID | Fréchet Inception Distance | ImageNet 生成质量主指标（越低越好） |
| Gflops | Giga floating-point operations | 本文缩放分析的核心复杂度度量 |

## 为什么重要

- **架构统一：** 证明扩散图像生成可与 ViT/语言 Transformer **同族缩放**，U-Net 跳跃连接并非必要条件（对后续 **VLA / WAM 动作 DiT 头** 是概念先例）。
- **可预测的 scaling law（图像域）：** 在控制训练步数下，**更高 Gflops**（更深更宽或更小 patch → 更多 token）→ **更低 FID**；参数量 alone 不足以解释（如 XL/8 vs XL/2）。
- **机器人侧读法：** 本站 [Diffusion Transformer 概念页](../concepts/diffusion-transformer.md) 与 [VLA](../methods/vla.md) 中的「DiT 动作专家」多指 **本条骨干 + flow/DDPM 目标** 的迁移，而非 ImageNet 类条件训练配方本身。
- **可复现：** [facebookresearch/DiT](https://github.com/facebookresearch/DiT) 提供 `sample.py` 与 XL/2 权重；与机器人专用 [DiT4DiT](./paper-dit4dit-video-action-model.md)（双 DiT 联合 VAM）区分命名。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 加州大学伯克利分校（UC Berkeley）；纽约大学（NYU）；Meta AI FAIR（Peebles 实习） |
| **出处** | ICCV 2023 Oral；arXiv:2212.09748（2022-12） |
| **任务** | Class-conditional ImageNet 256×256 / 512×512 生成 |
| **骨干** | 冻结卷积 VAE latent（Stable Diffusion 的 f8 VAE，256×256×3 → 32×32×4）+ **DiT**（patch embed + DiT blocks + adaLN-Zero） |
| **规模** | S/B/L/XL：12/12/24/28 层，隐藏维 384/768/1024/1152；XL 约 675M 参数、118.6 Gflops（p=2） |
| **训练** | AdamW，lr 1e-4 常数，无权重衰减，batch 256，仅水平翻转，EMA 0.9999；无需 warmup 与正则，未见 loss 尖峰 |
| **缩放网格** | 4 规模 × 3 patch（2/4/8）= 12 配置；主结果 **DiT-XL/2** |
| **开源** | **已开源** — [facebookresearch/DiT](https://github.com/facebookresearch/DiT)；权重 README 直链；[HF Space Demo](https://huggingface.co/spaces/wpeebles/DiT) |

## 核心原理

1. **Patchify：** 带噪 latent \(z_t\)（如 32×32×4）切成 patch，线性嵌入为长度为 \(T=(I/p)^2\) 的 token 序列；**patch 大小 \(p\)** 是调节 Gflops 的主旋钮（\(p\) 减半 → token 约 ×4 → Gflops 约 ×4）。
2. **DiT block（四种变体）：** 论文比较 in-context（t、c 作额外 token）、cross-attention（多一层交叉注意力，约 +15% Gflops）、adaLN（用 t 与 c 嵌入之和回归每层 LayerNorm 的 γ、β）和 **adaLN-Zero**（另回归残差前的缩放 α，MLP 零初始化使整块初始为恒等）。XL/2 在 400K 步的无引导 FID-50K：in-context 35.24、cross-attention 26.14、adaLN 25.21、adaLN-Zero 19.47，此后全文用 adaLN-Zero。adaLN 是唯一对所有 token 施加同一函数的条件机制。
3. **解码与输出：** 最后一个 block 后经（自适应）LayerNorm 与线性层，把每个 token 解码为 $p\times p\times 2C$ 张量（噪声 + 对角协方差），再重排回空间布局。
4. **训练目标：** 沿用 ADM/DDPM，$\epsilon_\theta$ 用 $L_{simple}=\|\epsilon_\theta(x_t)-\epsilon_t\|^2$ 训练，方差 $\Sigma_\theta$ 用完整 $L$ 训练；**CFG** 训练时随机将类标签换为 null embedding，采样时 $\hat\epsilon=\epsilon_\theta(x_t,\varnothing)+s\,(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing))$。
5. **采样：** 论文评测用 250 步 DDPM；项目页展示 DDIM 隐空间 walk 与 label embedding 插值。

### 流程总览

```mermaid
flowchart LR
  z["带噪 latent z_t"] --> patch["patchify → tokens"]
  t["时间步 t"] --> ada["AdaLN 调制"]
  c["类标签 c"] --> ada
  patch --> blocks["DiT Transformer 堆"]
  ada --> blocks
  blocks --> eps["预测噪声 ε"]
  eps --> samp["DDPM/DDIM 迭代"]
  samp --> dec["VAE 解码 → 图像"]
```

## 源码运行时序图

官方仓入口见 [sources/repos/facebookresearch-dit.md](../../sources/repos/facebookresearch-dit.md)：

```mermaid
sequenceDiagram
  autonumber
  actor Dev as 开发者
  participant Env as Conda DiT 环境
  participant Sample as sample.py
  participant Hub as dl.fbaipublicfiles.com
  participant Model as models.py DiT-XL/2
  participant VAE as 预置 LDM VAE
  Dev->>Env: conda env create -f environment.yml
  Dev->>Sample: python sample.py --image-size 512 --seed 1
  Sample->>Hub: 下载 DiT-XL-2-512x512.pt（若本地无）
  Sample->>Model: 加载权重 + CFG 采样循环
  Model->>VAE: 解码 latent → PNG
  Note over Dev,VAE: 训练路径：train.py + ImageNet + DDP（见 README）
```

## 评测与指标

| 模型 | 分辨率 | FID-50K | Inception Score | 去噪 Gflops |
|------|--------|---------|-----------------|-------------|
| DiT-XL/2（cfg=1.5，训练 7M 步） | 256×256 | **2.27** | 278.24 | 118.6 |
| DiT-XL/2（cfg=1.5，训练 3M 步） | 512×512 | **3.04** | 240.82 | 524.6 |
| 对照 LDM-4-G（cfg=1.5） | 256 | 3.60 | 247.67 | 约 104（正文 103.6） |
| 对照 ADM-G,ADM-U | 256 / 512 | 3.94 / 3.85 | — | 742 / 2813（ADM-U） |
| 对照 StyleGAN-XL | 256 | 2.30 | 265.12 | — |

- **Scaling：** 400K 步时 12 个配置的 Gflops 与 FID-50K 相关系数为 −0.93（Fig. 8）；DiT-S/2 与 DiT-B/4 Gflops 相近、FID 相近；保持模型大小只减小 patch 时参数量几乎不变（略降）而 FID 显著改善，故“算力而非参数量”是关键。
- **采样算力补不了模型算力：** L/2 用 1000 步（80.7 Tflops）FID-10K 为 25.9，劣于 XL/2 用 128 步（15.2 Tflops）的 23.7。
- **CFG：** 论文 2.27 对应 cfg=1.5，且**引导只施加在 latent 的前 3 个通道**（附录 A；四通道约 1.375 得 2.20）。此前页面写的“256 常用 4.0、512 常用 6.0”是官方仓库 `sample.py` 的采样设置，不是论文的评测设置，勿混用。
- **VAE 解码器：** original / ft-MSE / ft-EMA 解码器的 FID 为 2.46 / 2.30 / 2.27（Table 5），最终表用 ft-EMA；参数与 Gflops 不含 84M 参数的 VAE。
- **机器人 relevance：** 本文 **不** 含操控/导航 benchmark；对具身栈的价值在 **架构与缩放证据**，见 [diffusion-transformer](../concepts/diffusion-transformer.md)。

## 与其他工作对比

| 对照 | DiT（本文） | U-Net LDM / ADM | 机器人 [DiT4DiT](./paper-dit4dit-video-action-model.md) |
|------|-------------|-----------------|-----------------------------------------------------------|
| 骨干 | patch ViT + AdaLN | 卷积 U-Net + 空间注意力 | **双 DiT**（Video + Action）联合 flow matching |
| 条件 | 类标签 + 时间步 | 同类 + 多模态变体 | 视频隐状态 + 动作块 |
| 缩放度量 | **Gflops** 为主 | 参数量/Gflops 均有讨论 | 样本效率相对 Grounding/FLARE（机器人指标） |
| 开源 | facebookresearch/DiT | 各基线分散 | Mondo-Robotics/DiT4DiT |

## 结论

**DiT 把「Transformer 能否替代 U-Net 做扩散」答成肯定，并用 Gflops–FID 给出可复现的缩放曲线，成为后续文生图与具身 DiT 动作头的共同模板。**

- 选型时优先看 **目标 Gflops / token 数**，不要只看参数量（XL/8 反例）。
- 条件注入上 **AdaLN-Zero 残差调制** 是稳定训练的关键细节，迁移到动作 DiT 时值得保留。
- 复现论文图像结果走 `sample.py` + 官方权重；训练全量 ImageNet 需 DDP 与数据管线，与机器人小数据微调不是同一路径。
- 读 VLA 论文中的 DiT 头：数学仍是扩散/flow，**骨干结构** 可回溯本文。
- ImageNet 类条件 ≠ 语言/动作条件；跨模态需另设计 cross-attn 或观测 encoder（见 WAM/VLA 页）。
- 论文本身只讨论类条件与时间步条件，没有文本条件，也没有机器人实验；其结论外推到“VLA 动作头”属于本站推测，需以具体 VLA 论文核实。

## 工程实践

| 项 | 建议 |
|----|------|
| 最短验证 | `sample.py --image-size 256` 拉权重生成 grid |
| 算力规划 | patch 2 显著增 Gflops；高分辨率优先 LDM + 适中 patch |
| 与 diffusers | HF diffusers DiT pipeline 适合快速对比采样器 |
| 机器人 | 动作序列更短，常缩小 depth/width；延迟看 **每步注意力成本** |

## 局限与风险

- **二次注意力成本：** token 数随分辨率/patch 爆炸（512×512 的 XL/2 为 1024 token、524.6 Gflops）；机器人实时环需截断步数或蒸馏（推测）。
- **原文自述范围：** 仅在 ImageNet 类条件生成上验证；总结把文本到图像骨干（DALL·E 2、Stable Diffusion）和更大规模扩展列为未来工作。FID 对实现细节敏感，论文统一用 ADM 的 TensorFlow 评测套件。
- **类条件 ImageNet 配方** 不能直接套用到语言条件 VLA，需重训条件接口。
- VAE 质量与 latent 尺度绑定；换 VAE 须重训 DiT。

## 关联页面

- [Diffusion Transformer（概念）](../concepts/diffusion-transformer.md)
- [扩散模型](../concepts/diffusion-model.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [VLA](../methods/vla.md)
- [DiT4DiT（机器人双 DiT VAM）](./paper-dit4dit-video-action-model.md)
- [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)（Components 分组 #086）

## 参考来源

- [DiT 论文归档（arXiv:2212.09748）](../../sources/papers/peebles_dit_arxiv_2212_09748.md)
- [DiT 项目页（wpeebles.com）](../../sources/sites/dit-wpeebles-com.md)
- [facebookresearch/DiT 仓库归档](../../sources/repos/facebookresearch-dit.md)

## 推荐继续阅读

- [ICCV 2023 OpenAccess 论文页](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)
- [Hugging Face DiT Space](https://huggingface.co/spaces/wpeebles/DiT)
- [Latent Diffusion Models（Rombach et al.）](https://arxiv.org/abs/2112.10752) — DiT 训练的 latent 框架
