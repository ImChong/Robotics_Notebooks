---
title: VLA和世界模型的宝藏教程！入门到进阶的阅读路线图，附必读论文
author: 具身智能研究室
date: "2026-09-02 17:30:00"
source: "https://mp.weixin.qq.com/s/fNAyDttYIs5kzTQHwxc5Pw"
---

# VLA和世界模型的宝藏教程！入门到进阶的阅读路线图，附必读论文

10 篇必读 VLA 及相关基础论文 + 4 篇世界模型补充，从奠基之作到最新开源模型，每篇附"为什么读"和"核心收获"。📚 **想查看更完整的具身智能论文分类与整理？** 欢迎访问 Embodied-AI-Paper-Analysis — 覆盖 VLA、强化学习、世界模型等 7 大方向的论文体系化梳理，按顶会分类、带 venue tier 标注。

## 里程碑论文

### 01 RT-1: Robotics Transformer 1

**论文:** *RT-1: Robotics Transformer for Real-World Control at Scale* (Google DeepMind, 2022)

**arXiv:** 2212.06817

**代码:** google-research/robotics\_transformer

方法框架图

![RT-1 方法框架图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXO25JqsTVPfreYF3ibX8fpMJBxn9KQEbIiclDwtbibQicpMvvcQkbz7ek3qiczu2OxogiaibKzLoNWyChoWObrClHFUru69pv2dN2bIY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)图：RT-1 从语言指令与历史图像到离散动作的完整架构。**为什么读：**VLA 领域的开山之作。首次证明 Transformer 可以在大规模真实机器人数据上训练，实现端到端控制。

核心架构

- 输入：6 张历史图像（FiLM 条件化语言指令）+ 自然语言指令
- 主干：EfficientNet-B3 视觉编码器 → TokenLearner 压缩 → Transformer Decoder
- 输出：256 个离散动作 bin（自回归生成）
- 数据：130k 条演示，700+ 任务

核心收获

- 历史帧（temporal context）对操作任务至关重要
- TokenLearner 将视觉 token 从 6k+ 压缩到 81 个，大幅降低计算量
- 大规模数据 + 简单架构 > 小数据 + 复杂架构

### 02 RT-2: Vision-Language-Action Models

**论文:** *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (Google DeepMind, 2023)

**arXiv:** 2307.15818

**代码:** 官方未开源完整训练代码，推理参考社区实现

方法框架图

![RT-2 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEVoMmAexxtIXAJF4DzRrT48WTK8pzEa1tkDSVAUnqhFuXSFAYPI61RAV93zoult7p0MWsQPTKlHZXrY0iaoPTfw7u11mVxqthbQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)图：RT-2 将机器人动作表示为文本 token，与互联网视觉-语言数据联合微调。**为什么读：****VLA 的命名来源。**首次将预训练 VLM（PaLI-X / PaLM-E）的语义知识迁移到机器人控制。

核心思想

- 将机器人动作表示为文本 token（如 `"1 128 91 241 5 1"`）
- 直接微调预训练 VLM，使其输出这些"动作文本"
- 同时保留 VLM 的语义推理能力（可解释符号、推理物体关系）

关键创新

- **Co-training:** 在机器人数据 + 视觉-语言数据上联合训练，防止灾难性遗忘
- 模型可执行**推理链：**"杯子是易碎的 → 需要轻拿轻放 → 减小夹持力"

核心收获

- VLM 的互联网知识可以通过微调迁移到物理控制
- 动作离散化是将连续控制问题转化为语言建模问题的桥梁
- 泛化能力远超纯 BC 方法（可识别训练时未见的物体）

### 03 OpenVLA: An Open-Source Vision-Language-Action Model

**论文:** *OpenVLA: An Open-Source Vision-Language-Action Model* (Stanford / UC Berkeley / MPI, 2024)

**arXiv:** 2406.09246

**代码:** openvla/openvla ⭐ 强烈推荐

方法框架图

![OpenVLA 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUHuouOIh6mp1n8PCptZ6SaM5I8v7pVNbbNIgKhE3lc2xAYZoFy89HcmFvhz8zre0ArKxiaTTLBl1e9xUGumXvSkIiakYHxJ7pWo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)图：OpenVLA 的双视觉编码器、投影层、Llama 2 主干与动作反分词器。**为什么读：****当前最活跃的开源 VLA 项目。**7B 参数，性能接近 RT-2-X（55B），训练成本仅 $30k。

核心架构

- 视觉编码器：**DINOv2** + **SigLIP**（双塔融合）
- 语言主干：**Llama 2**（7B）
- 动作表示：将 7 维动作的各维离散化为 256 个 bin，并作为 token 进行自回归预测
- 训练数据：Open X-Embodiment（970k 条轨迹）+ 内部数据

关键设计

- DINOv2 提供空间几何理解，SigLIP 提供语言对齐
- 使用 **Llama 2** 而非 T5，利用其强大的推理能力
- 支持多图像输入（单臂 / 双臂 / 腕部相机）

核心收获

- 开源模型可以达到闭源 SOTA 的 85%+ 性能
- 统一的动作 token 表示让通用语言模型可直接用 next-token prediction 学习机器人控制
- 预训练后微调是关键：在目标机器人上微调 5k-10k 步即可适配

快速上手

bash

```
pip install openvla
# 或使用 transformers
from transformers import AutoModelForVision2Seq, AutoProcessor
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
```

### 04 π0 (pi-zero): A Vision-Language-Action Flow Model

**论文:** *π0: A Vision-Language-Action Flow Model for General Robot Control* (Physical Intelligence, 2024)

**arXiv:** 2410.24164

**代码:** physical-intelligence/pi0

方法框架图

![pi0 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEX5eWkVR0VsjONZh66UmLUchgR4icvc55hKYNI9jzRibsbicQJgc5vOuU1r6iczCIE8t2JkhAkibIQ7GibFgrBh3WCfDuspPAoq8gXTc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)图：π0 的预训练数据混合、VLM 主干和基于流匹配的 Action Expert。**为什么读：**使用**流匹配（Flow Matching）**生成动作，在精细操作任务上表现出色（叠衣服、装袋等）。

核心架构

- 基于 **Diffusion / Flow Matching** 生成动作
- 预训练 VLM 主干（类似 PaliGemma）
- 关键创新：**Action Expert** — 冻结 VLM，只训练专门的动作生成模块

关键设计

- Flow Matching 比传统 Diffusion 更快（单次前向 vs 多步去噪）
- 支持高频控制（50Hz）
- 在混合数据（操作 + 导航 + 移动操作）上训练

核心收获

- 扩散/流模型适合生成平滑、多峰的动作分布
- Action Expert 设计实现了"语义理解冻结，动作生成可训练"
- 高频控制需要高效的推理 pipeline

### 05 Octo: An Open-Source Generalist Robot Policy

**论文:** *Octo: An Open-Source Generalist Robot Policy* (Berkeley / Stanford / Google, 2024)

**arXiv:** 2405.12213

**代码:** octo-models/octo

方法框架图

![Octo 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWRT1f5tt7xzFWfmNyXnVGjP1ZngicnThzEhoLicAJibxGoLSXF1PGXVMBJHoQZS88RKibOQeukFQXhAALQGBeS4Z7yftJ8l08kibWs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)图：Octo 的任务/观测 token、读出 token、动作头及微调时的可扩展注意力结构。**为什么读：****最灵活的 VLA 框架。**支持任意观察（图像、点云、关节角）、任意任务、任意机器人。

核心架构

- 基于 **Transformer**，但比 RT 系列更灵活
- **读取器-写入器注意力（Read-Write Attention）：**

- 读取器：处理输入观察（可多模态）
- 写入器：生成动作 token

- 支持 **Goal Conditioning：**目标图像 + 语言指令

关键设计

- 统一所有输入为 token 序列，无需固定输入格式
- 支持多种动作表示（关节角、末端位姿、增量）
- 轻量：27M 参数即可工作

核心收获

- 架构灵活性比参数量更重要（27M Octo vs 7B OpenVLA 各有适用场景）
- Goal Image Conditioning 对需要目标状态的任务非常有用
- 多机器人联合训练需要仔细处理数据格式统一

## 重要扩展论文

### 06 Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

**论文:** *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (Columbia / MIT, 2023)

**arXiv:** 2303.04137

**代码:** real-stanford/diffusion\_policy

方法框架图

![Diffusion Policy 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWQz3h8ITnzkVrLz8YKEoiambnRhPOtqRARWCiccq5ibFibuulsicKic1uav8iakSoGBsZ3fiaQFiccpeCepz4bVkHwjMG0OzY3YaibQ2JsM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)图：Diffusion Policy 的闭环动作序列生成以及 CNN/Transformer 两种去噪网络。**为什么读：**虽然不是严格意义上的 VLA（没有语言输入），但扩散策略被 π0 等 VLA 模型采用作为策略头。

**核心思想：**将动作生成建模为去噪过程：

python

```
# 训练：向真实动作加噪，训练去噪网络
noise = torch.randn_like(action)
noisy_action = sqrt(alpha) * action + sqrt(1-alpha) * noise
predicted_noise = denoiser(noisy_action, obs, timestep)
loss = MSE(predicted_noise, noise)

# 推理：从纯噪声逐步去噪
action = torch.randn(T, action_dim)
for t in reversed(range(T)):
    action = denoiser.step(action, obs, t)
```

核心收获

- 扩散模型可以表示多峰动作分布（一个场景有多个可行解）
- 比 GMM（高斯混合模型）和 VAE 更适合动作生成
- 去噪迭代次数影响速度与质量的 trade-off

### 07 CLIP: Learning Transferable Visual Models From Natural Language Supervision

**论文:** *Learning Transferable Visual Models From Natural Language Supervision* (OpenAI, 2021)

**arXiv:** 2103.00020

**代码:** openai/CLIP

方法框架图

![CLIP 方法框架图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWLeiaOcRYv2s6eyAXTGE4SwncRFbINH8Jgy7MzLsgoIP0wzSluQKibD4k9UUA5IkyhYHFu2DrsIyOFXH9v1Q703HcT1Ucs2oQVU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)图：CLIP 的图文对比预训练与零样本分类流程。**为什么读：**VLA 的**视觉-语言对齐基础。**几乎所有 VLA 都使用 CLIP 或其变体作为视觉编码器。

**核心思想：**对比学习，让匹配的图像-文本对在嵌入空间靠近：

python

```
# 图像编码器 + 文本编码器
image_features = image_encoder(image)   # [N, D]
text_features = text_encoder(text)       # [N, D]

# 对比损失
logits = image_features @ text_features.T / temperature
labels = arange(N)
loss = cross_entropy(logits, labels) + cross_entropy(logits.T, labels)
```

核心收获

- 视觉-语言对齐是 VLA 的基石
- CLIP 的 zero-shot 能力使模型能识别训练时未见的物体
- 后来的 DINOv2、SigLIP 都在 CLIP 基础上改进

### 08 ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware

**论文:** *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Stanford, 2023)

**arXiv:** 2304.13705

**代码:** tonyzhaozh/act

方法框架图

![ACT 方法框架图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXA6perKW9Fia1yz0LjQFaicdHP2xLNqeAhuHqmyEVDpN9iacTbKYEpzRlEh6WlLrMtz9trAkyVz71EoGw4NF4OpH5T3DVMuZ0h5E/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)图：ACT 的 CVAE 编码器、多视角视觉编码与 Action Chunk Transformer 解码器。**为什么读：****最实用的入门项目。**只用 $2k 的 ALOHA 硬件，实现双手精细操作。

核心架构

- 基于 Transformer 的编码器-解码器
- **Action Chunking with Transformer (ACT)：**一次性预测未来 K 步动作
- **CVAE：**生成多样化的动作 chunk

核心收获

- Action Chunking 大幅降低推理频率（从 50Hz 降到 5Hz）
- 低成本硬件 + 简单算法可以实现令人惊讶的操作精度
- 双手协调需要同时建模两个臂的动作相关性

### 09 SPOC: Imitating Shortest Paths for Sim-to-Real Navigation and Manipulation

**论文:** *Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World* (Allen Institute for AI, 2023)

**arXiv:** 2312.02976

**代码:** allenai/spoc-robot-training

方法框架图

![SPOC 方法框架图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWia46FdNNGgolghclLzNBalOzkc6cnOZzx9Z3LIiaT2iajoicDuiaYzRTBiayeyPfH97OmJBAwQFlOK4DicInCNzCOzTog2A9ljxhwjc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)图：SPOC 的目标条件视觉编码器与基于 Transformer 的动作解码器。**为什么读：**展示如何在大规模程序化仿真住宅中模仿最短路径专家，训练可零样本迁移到现实环境的移动导航与抓取策略。

核心思想

- 从 ProcTHOR 生成的程序化住宅和最短路径专家轨迹中进行模仿学习
- 用目标条件视觉编码器融合语言目标、导航相机与操作相机观测
- 通过 Transformer 动作解码器结合当前/历史观测和历史动作预测控制命令

核心收获

- 便宜的仿真规划器可以规模化生成长时程具身训练数据
- 大规模环境多样性与视觉增强对 sim-to-real 迁移至关重要
- 同一目标条件策略可统一处理导航、寻物、抓取等任务

### 10 DINOv2: Learning Robust Visual Features without Supervision

**论文:** *DINOv2: Learning Robust Visual Features without Supervision* (Meta, 2023)

**arXiv:** 2304.07193

**代码:** facebookresearch/dinov2

方法框架图

![DINOv2 数据处理框架图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEV9x2CIlSTIJIXMLpE0B4wEvcHXN4pszo4pAfWh4Qf5ML4tBl54fXo3RKq0Bk0KlMZgbOrHFlkRnmMDo71n61PjGLgRPy18qWg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)图：DINOv2 的 LVD-142M 数据构建流程，包括嵌入、去重与自监督检索。**为什么读：**OpenVLA 使用的视觉编码器之一。自监督预训练，空间理解能力远超 CLIP。

核心思想

- 自蒸馏（self-distillation）：学生网络预测教师网络的输出
- 使用 DINO 损失 + iBOT 掩码预测
- 在大规模图像数据集上预训练（142M 图像）

核心收获

- 自监督视觉特征在几何/空间任务上优于 CLIP
- DINOv2 的 attention map 可以显示模型"在看哪里"
- 与 SigLIP 互补：DINOv2 提供空间理解，SigLIP 提供语言对齐

## 世界模型补充（VLA 融合方向）

随着项目从纯 VLA 扩展为"VLA + RL + 世界模型"三大支柱，以下补充论文聚焦世界模型如何直接服务于机器人操作与 VLA 系统。

### 11 LaDi-WM: Latent Diffusion World Model for Predictive Manipulation

**论文:** *LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation* (国防科大 / 北京大学 / 深圳大学, CoRL 2025)

**arXiv:** 2505.11528

**项目页:** LaDi-WM Project

方法框架图

![LaDi-WM 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUvmQwnktCVnBqfTH40VjJxru5U3cgZ8oloWiaqf1pyjzwMicM9XHPnpAAR4dxszNvNs4oFiaplkXslBGqO7jVwHf74T4KJGJ4JHo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)图：LaDi-WM 的隐空间交互扩散世界模型与预测式操作策略。**为什么读：**世界模型与 VLA 融合的最新代表作。首次将 Latent Diffusion 引入机器人操作，在隐空间建模动力学，实现"世界模型预测 → 迭代优化策略"的闭环。

核心架构

- **双塔隐空间：**DINOv2（几何）+ SigLIP（语义）构建通用隐表示
- **交互扩散：**几何与语义表征在扩散过程中交互，学习联合动力学
- **迭代策略优化：**用 WM 未来预测多次引导策略，逐步降低动作熵

核心收获

- 隐空间扩散比像素级预测更适合机器人操作（计算高效 + 跨场景泛化强）
- 世界模型泛化能力优于策略模型：跨数据集（LIBERO → CALVIN）零样本迁移提升显著
- 少量轨迹（10 条）即可达到 68.7% 成功率，对真实机器人数据稀缺场景极具价值

VLA 关联

- LaDi-WM 的隐空间设计（DINOv2 + SigLIP）与 OpenVLA 视觉编码器一致，可直接作为 VLA 的"预测模块"
- 对应融合方式：世界模型作为规划器（本文档 docs/07 第 4.3 节）
- 完整解读见 `docs/07-world-models-for-vla.md`

### 12 DreamDojo: Generalist Robot World Model from Large-Scale Human Videos

**论文:** *DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos* (UT Austin / NVIDIA, ICML 2026)

**arXiv:** 2602.06949

**代码:** NVIDIA/DreamDojo (Apache-2.0)

方法框架图

![DreamDojo 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEVOkRjQRYsGDaz84nfOgX7HGwPd1E6SicYrSUCEMIWibVaoibtNiaAElWG6SCYVehVxcfkI8LiaZ5eEMgYx0TEakwJtNmMntOKBKXibg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)图：DreamDojo 从人类视频预训练、机器人后训练到自回归蒸馏和下游应用的总体流程。**为什么读：**从海量人类视频预训练通用机器人世界模型的代表性工作，引入 latent action 解决无动作标签数据训练问题。

核心架构

- **Latent Action Model：**隐式编码人类视频中的动作信息
- **蒸馏加速：**约 10 FPS 实时交互
- **Post-training：**少量机器人数据即可迁移

核心收获

- 世界模型可以从人类视频规模化预训练，无需昂贵的机器人演示数据
- latent action 是跨 embodiment 动作表示的重要思路
- 蒸馏方案在生成质量和推理速度间取得平衡

VLA 关联

- 直接对应 VLA 的数据瓶颈问题
- 与你当前研究的"人类数据→机器人控制"方向高度契合
- 完整解读见 `docs/07-world-models-for-vla.md`

### 13 RISE: Self-Improving Robot Policy with Compositional World Model

**论文:** *RISE: Self-Improving Robot Policy with Compositional World Model* (OpenDriveLab, RSS 2026)

**arXiv:** 2602.11075

**代码:** OpenDriveLab/RISE

方法框架图

![RISE 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUWTcwz0s4iarIGXjIhTnxoEkqskew4SaTM4otiaQIicrfnbVJicOOWcfcCNfx62Wl4t1G5LaibGiccCw3kicJ7ygp10Rpr2MuDBVh4Tc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)图：RISE 将真实世界 RL 转换为组合式世界模型中的想象强化学习。**为什么读：**世界模型如何真正帮助机器人策略变强的工程化实现，提供完整的 WM + RL 闭环代码。

核心架构

- **Dynamics Model：**可控环境动力学
- **Progress/Value Model：**评估任务进度
- **Imagination RL：**在想象世界中训练策略
- **PiPER 部署：**想象训练 → 真实机器人

核心收获

- 组合式世界模型分离 dynamics 和 value，训练更稳定
- imagination rollout 可以替代真实环境的 RL 交互，大幅降低硬件成本
- 完整代码链路（offline policy → online RL → real robot）值得复现

VLA 关联

- 对应"高质量重定向数据 → 世界模型 → RL 自提升"的完整路线
- 与你当前的重定向项目可以自然延伸结合
- 完整解读见 `docs/07-world-models-for-vla.md`

### 14 PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation

**论文:** *PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation* (NVIDIA / Stanford, CVPR 2026 Highlight)

**arXiv:** 2601.03782

**代码:** NVlabs/PointWorld

方法框架图

![PointWorld 方法框架图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEVAqGVZQaHZcNibiaysDvbM4cggxJZptiaDoibbdHLAINLEetwBDTVUargpyRFPjJ3SytLLhkJOsqwlia9hVPVNOpTWvKgKwhLYA8Wo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)图：PointWorld 将关节动作转为机器人点流，与 RGB-D 场景点云统一建模并预测全场景 3D 点流。**为什么读：**3D 跨本体世界模型的开创性工作，与你当前跨形态重定向研究直接相关。

核心架构

- **3D Point Flow：**统一表示 world state + action
- **跨 embodiment：**不依赖特定机器人的关节空间
- **MPC 实时规划：**0.1 秒推理速度

核心收获

- 3D point flow 比 RGB 视频更适合跨本体世界模型
- 减少对特定 action representation 的依赖
- 200 万条轨迹的大规模训练验证了 3D 世界模型的可扩展性

VLA 关联

- 3D point flow 可作为跨本体世界模型的统一表示
- 未来可结合：3D point flow 作为中间表示 → 机器人适配器 → 具体机器人动作
- 完整解读见 `docs/07-world-models-for-vla.md`

## 阅读路线图

```
入门路线：
CLIP → RT-1 → RT-2 → OpenVLA
        ↓
    Diffusion Policy → π0
        ↓
    ACT（动手实践）

进阶路线：
OpenVLA 源码精读 → Octo 架构设计 → π0 Flow Matching
```

## 论文资源汇总

| 论文 | arXiv | 代码 | 难度 |
| --- | --- | --- | --- |
| CLIP | 2103.00020 | GitHub | ★☆☆ |
| DINOv2 | 2304.07193 | GitHub | ★★☆ |
| Diffusion Policy | 2303.04137 | GitHub | ★★☆ |
| RT-1 | 2212.06817 | GitHub | ★★☆ |
| ACT | 2304.13705 | GitHub | ★★☆ |
| RT-2 | 2307.15818 | 社区实现 | ★★★ |
| SPOC | 2312.02976 | GitHub | ★★★ |
| Octo | 2405.12213 | GitHub | ★★★ |
| OpenVLA | 2406.09246 | GitHub | ★★★ |
| π0 | 2410.24164 | GitHub | ★★★★ |
| LaDi-WM | 2505.11528 | Project | ★★★★ |
| DreamDojo | 2602.06949 | GitHub | ★★★★ |
| RISE | 2602.11075 | GitHub | ★★★★ |
| PointWorld | 2601.03782 | GitHub | ★★★★ |
