# Wan: Open and Advanced Large-Scale Video Generative Models（arXiv:2503.20314）

> 来源归档（ingest）

- **标题：** Wan: Open and Advanced Large-Scale Video Generative Models
- **类型：** paper / video foundation model / DiT / VAE / T2V / I2V
- **arXiv：** <https://arxiv.org/abs/2503.20314>（PDF：<https://arxiv.org/pdf/2503.20314.pdf>）
- **官方站点：** <https://wan.video>
- **代码（2.1）：** <https://github.com/Wan-Video/Wan2.1>
- **代码（2.2 升级线）：** <https://github.com/Wan-Video/Wan2.2>
- **权重组织：** HF <https://huggingface.co/Wan-AI/>；ModelScope <https://modelscope.cn/organization/Wan-AI>
- **作者：** Wan Team 等（报告署名含 Team Wan 及大量贡献者；通义 / 阿里巴巴主导）
- **机构：** 阿里巴巴（Alibaba）
- **入库日期：** 2026-07-23
- **一句话说明：** 开源大规模视频基础模型技术报告：DiT + 新型 **Wan-VAE**、可扩展预训练与数据策展；提供 **1.3B / 14B** 等规模，覆盖 T2V / I2V 等多任务，并在开源与商业对照上报告领先表现。后续 **Wan2.2**（MoE、美学、更大数据、TI2V-5B）与 **Wan-Move / Wan-Fun-Control** 等可控衍生均以此族为骨干——是本库多篇机器人视频 WM 的 **上游先验**。

## 开源状态（站点 + 仓库核查，2026-07-23）

- **已开源：** [`Wan-Video/Wan2.1`](https://github.com/Wan-Video/Wan2.1) 与 [`Wan-Video/Wan2.2`](https://github.com/Wan-Video/Wan2.2)（均为 **Apache-2.0**）+ HF/ModelScope 权重；`generate.py` 支持 T2V / I2V / FLF2V / VACE 等；已集成 Diffusers / ComfyUI。技术报告对应主发布线为 **Wan2.1**；**Wan2.2** README 仍指向同一 arXiv 论文并声明为 major upgrade。

## 摘要级要点

- **定位：** 综合开源视频基础模型套件，非单任务机器人论文；但对具身侧意义在于提供 **可微调、可控扩展** 的强视觉动力学先验。
- **特征（报告口径）：** Leading Performance（14B + 大规模图文/视频数据）；Comprehensiveness（1.3B 与 14B、多下游）；高效 VAE；可消费级 GPU 跑小模型（如 T2V-1.3B ~8.19 GB VRAM）。
- **机器人谱系接点：**
  - [Wan-Move](../../wiki/entities/paper-wan-move.md) — 轨迹可控 I2V（微调 Wan-I2V-14B）
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — Wan2.2-Fun-Control + LoRA
  - 本库多页引用 Wan2.1/2.2 作视频骨干（ABot、τ₀、TouchWorld、PanoWorld 等）

## 核心论文摘录（MVP）

### 1) 开源视频基础模型定位

- **链接：** Abstract；Introduction
- **摘录要点：** 在主流 DiT 范式上，用新 VAE、可扩展预训练、大规模数据与自动评测推动开源视频生成；14B 展示数据/模型缩放；覆盖多下游应用。
- **对 wiki 的映射：**
  - [Wan](../../wiki/entities/paper-wan-video.md) — 基础模型实体。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 视频先验来源。

### 2) 工程交付面（以官方仓为准）

- **链接：** Wan2.1 / Wan2.2 README（非 PDF 专节）
- **摘录要点：** `generate.py` 统一入口；T2V-1.3B 可消费级推理；I2V-14B 为 Wan-Move 等微调起点；Wan2.2 引入 MoE 分时专家与更大美学/运动数据。
- **对 wiki 的映射：**
  - [`sources/repos/wan2.1.md`](../repos/wan2.1.md) — 运行入口。
  - [Wan-Move](../../wiki/entities/paper-wan-move.md) / [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — 下游可控适配。

### 3) 与机器人视频 WM 的边界

- **链接：** 社区生态（README Community Works）
- **摘录要点：** 官方列举 Wan-Move、驾驶 WM、动画等社区延伸；机器人侧需另加 **动作 / 掩码 / IR** 条件与真机数据，不能把通用 I2V 直接当闭环物理引擎。
- **对 wiki 的映射：**
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 「视频先验 ≠ 即可部署仿真」。
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) — 另一条（SVD）动作条件 WM 对照。

## BibTeX

```bibtex
@article{wan2025,
  title   = {Wan: Open and Advanced Large-Scale Video Generative Models},
  author  = {{Wan Team} and others},
  journal = {arXiv preprint arXiv:2503.20314},
  year    = {2025}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-wan-video.md`](../../wiki/entities/paper-wan-video.md)
- 代码归档：[`sources/repos/wan2.1.md`](../repos/wan2.1.md)
- 站点：[`sources/sites/wan-video.md`](../sites/wan-video.md)
- 互链：[Wan-Move](../../wiki/entities/paper-wan-move.md)、[Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)
