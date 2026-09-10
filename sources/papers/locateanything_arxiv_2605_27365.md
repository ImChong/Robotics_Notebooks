# LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding（arXiv:2605.27365）

> 来源归档（ingest）

- **标题：** LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
- **中文译名：** LocateAnything：基于并行框解码的快速高质量视觉-语言定位
- **类型：** paper / vlm / visual-grounding / object-detection / open-vocabulary
- **arXiv：** <https://arxiv.org/abs/2605.27365>
- **项目页：** <https://research.nvidia.com/labs/lpr/locate-anything/>
- **PDF：** <https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf>
- **模型：** <https://huggingface.co/nvidia/LocateAnything-3B>
- **Demo：** <https://huggingface.co/spaces/nvidia/LocateAnything>
- **代码：** <https://github.com/NVlabs/Eagle/tree/main/Embodied> — 归档见 [`sources/repos/eagle_embodied_locateanything.md`](../repos/eagle_embodied_locateanything.md)
- **数据：** <https://huggingface.co/datasets/NVEagle/LocateAnything-Data>
- **作者：** Shihao Wang, Shilong Liu, Yuanguo Kuang, Xinyu Wei, Yangzhou Liu, Zhiqi Li, Yunze Man, Guo Chen, Andrew Tao, Guilin Liu, Jan Kautz, Lei Zhang, Zhiding Yu
- **机构：** 英伟达（NVIDIA）等（作者单位含香港理工大学、普林斯顿、南京大学、UIUC 等）
- **入库日期：** 2026-09-10
- **一句话说明：** 提出 **Parallel Box Decoding（PBD）**，把框/点作为原子单元一步并行解码，在统一 VLM 框架下同时提升定位精度与吞吐；配套 **LocateAnything-Data**（12M 图、138M 查询、785M 框）与 **3B** 权重已公开。

## 开源状态（步骤 2.5，2026-09-10）

| 资源 | 状态 | 说明 |
|------|------|------|
| 推理/评测代码 | **已开源** | [NVlabs/Eagle/Embodied](https://github.com/NVlabs/Eagle/tree/main/Embodied)（代码 Apache 2.0） |
| 权重 | **已发布** | [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)（**NVIDIA License**，非商业研究） |
| 训练数据 | **已发布** | [NVEagle/LocateAnything-Data](https://huggingface.co/datasets/NVEagle/LocateAnything-Data)（~2.41 TB；部分子集需上游媒体 hydrate） |
| 训练全流程 | **部分** | README 含训练/评测入口；大规模数据需 Energon + 子集下载脚本 |

## 核心摘录

### 1. 问题与 PBD

- 常见 VLM grounding 把 2D 框 **序列化成多个 1D 坐标 token**，逐 token 自回归解码 → 与框几何耦合结构不匹配，且推理成为瓶颈。
- **PBD**：把 **bounding box / point** 当作 **定长原子单元**，在 **单步** 预测完整坐标集 \((x_1,y_1,x_2,y_2)\)，保持框内几何一致性并解锁并行。

### 2. 架构与推理模式

- **骨干：** Moon-ViT 视觉编码器 + **Qwen2.5-3B-Instruct** 语言解码器 + MLP projector；原生分辨率视觉 token。
- **Fast Mode（MTP）：** 并行预测整框，吞吐最高（论文 **15.3 BPS**）。
- **Slow Mode（NTP）：** 坐标 token 自回归，稳定性上限（COCO ablation **F1 52.1**）。
- **Hybrid Mode（默认）：** 先 Fast，遇 **格式不规则** 或 **空间歧义** 时对问题 block **回退 NTP 重解码**；论文主结果 **12.7 BPS**，兼顾鲁棒。

### 3. LocateAnything-Data

- **12M** 唯一图像；**138M** 语言查询；**785M** 标注框/点。
- 域覆盖：通用 OD（66.9% 查询）、GUI grounding（16.5%）、指代表达（7.3%）、OCR（3.6%）、版式（3.5%）、点定位（2.2%）。

### 4. 主要结果（Hybrid，单 H100，3B）

| 对比/基准 | 要点 |
|-----------|------|
| **吞吐** | **12.7 BPS** vs Qwen3-VL **1.1 BPS**、Rex-Omni **5.0 BPS**（同量级叙述） |
| **LVIS / COCO** | mean F1 **+3.8% / +1.8%** vs Rex-Omni；LVIS IoU=0.95：**31.1 vs 20.7** |
| **Dense200 / VisDrone** | mean F1 **58.7 / 39.9** vs Rex-Omni **58.3 / 35.8** |
| **ScreenSpot-Pro** | mean F1 **60.3** SOTA（GUI grounding） |
| **DocLayNet / M6Doc** | **76.8 / 70.1** mean F1 |
| **HumanRef / RefCOCOg** | **78.7** mean F1 等 |

## 对 wiki 的映射

- [`wiki/entities/paper-locateanything.md`](../../wiki/entities/paper-locateanything.md) — 论文实体主页
- [`wiki/queries/robot-perception-stack-selection-loop.md`](../../wiki/queries/robot-perception-stack-selection-loop.md) — 开放词汇 2D 定位层选项
- [`wiki/queries/object-detection-model-selection.md`](../../wiki/queries/object-detection-model-selection.md) — 检测 vs VLM grounding 选型
