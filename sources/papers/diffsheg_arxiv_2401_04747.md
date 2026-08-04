# diffsheg_arxiv_2401_04747

> 来源归档（ingest）

- **标题：** DiffSHEG: A Diffusion-Based Approach for Real-Time Speech-driven Holistic 3D Expression and Gesture Generation
- **类型：** paper
- **作者：** Junming Chen, Yunfei Liu, Jianan Wang, Ailing Zeng, Yu Li, Qifeng Chen（HKUST + IDEA；Chen 实习于 IDEA）
- **arXiv：** <https://arxiv.org/abs/2401.04747>（2024-01）
- **PDF：** <https://arxiv.org/pdf/2401.04747>
- **会议：** CVPR 2024
- **代码：** <https://github.com/JeremyCJM/DiffSHEG>
- **项目页：** <https://jeremycjm.github.io/proj/DiffSHEG/>
- **视频：** <https://www.youtube.com/watch?v=HFaSd5do-zI>
- **入库日期：** 2026-07-31
- **一句话说明：** 用统一扩散 + UniEG-Transformer（表情→手势单向条件流）联合建模共语 3D 表情与手势分布，并以 FOPPAS（outpainting + DDIM）实现任意长、实时（~30+ FPS）流式生成；BEAT / SHOW 上 SOTA，用户研究主导偏好。

## 核心论文摘录（MVP）

### 1) 问题与主张（Abstract / Intro）

- **链接：** <https://arxiv.org/abs/2401.04747>
- **核心贡献：** 先前工作多单独做手势或表情；联合方案常拆开生成或确定性多任务解码，难匹配表情–手势联合分布与 many-to-many 语音→手势映射。DiffSHEG 是**显式建模联合分布**的统一扩散框架，并用 FOPPAS 解决扩散任意长实时采样。
- **对 wiki 的映射：**
  - [DiffSHEG 实体](../../wiki/entities/paper-diffsheg.md)
  - [扩散运动生成](../../wiki/methods/diffusion-motion-generation.md)

### 2) UniEG-Transformer — 表情→手势单向信息流

- **链接：** 论文 §3.3；项目页 Framework
- **核心贡献：** Mel-Spectrogram + 冻结 **HuBERT**；Motion-Speech Fusion Residual（LN+MLP 残差，通道拼接对齐时序）；Style-aware Transformer（AdaIN 注入 person ID + 扩散步 \(t\)，线性注意力）。去噪步 \(t\) 上把预测表情 \(\hat{x}_{0(t)}^{E}\) **detach** 后条件到手势分支，避免手势梯度干扰唇形/表情映射。
- **对 wiki 的映射：**
  - [扩散模型](../../wiki/concepts/diffusion-model.md)
  - [DiffSHEG 仓库](../../sources/repos/diffsheg.md)

### 3) FOPPAS — 任意长实时采样

- **链接：** 论文 §3.5；项目页 Arbitrary-long Sampling
- **核心贡献：** 训练不依赖前帧条件；测试时用 Repaint 式 **outpainting** 在重叠帧上固定上一 clip 尾部、生成剩余帧；**DDIM 25 步**替换 1000 步 DDPM；末两步重叠区线性 blending。报告 RTX 3090 约 **31.5 FPS**（含音频编码）。
- **对 wiki 的映射：**
  - [扩散运动生成](../../wiki/methods/diffusion-motion-generation.md)（长序列 / 流式采样对照）

### 4) 数据、评测与开源

- **链接：** 论文 §4；<https://github.com/JeremyCJM/DiffSHEG>
- **核心贡献：** BEAT（15 fps，34 帧窗）与 SHOW（SMPLX，30 fps，88 帧窗）。主指标含 FMD/FED/FGD、BA、Div、SRGR/PCM；用户研究（22 人）在 realism / sync / diversity 上主导偏好。官方仓含训练、自定义 wav 推理与 Google Drive checkpoint。
- **对 wiki 的映射：**
  - [DiffSHEG 仓库归档](../../sources/repos/diffsheg.md)
  - [DiffSHEG 项目页归档](../../sources/sites/diffsheg.md)
  - [Semantic Co-Speech Gesture（PNB）](../../wiki/entities/paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md)（共语手势→真机链路对照）

## 对 wiki 的映射（汇总）

- 主实体：[paper-diffsheg](../../wiki/entities/paper-diffsheg.md)
- 方法：[diffusion-motion-generation](../../wiki/methods/diffusion-motion-generation.md)
- 概念：[diffusion-model](../../wiki/concepts/diffusion-model.md)
- 对照：[paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti](../../wiki/entities/paper-notebook-semantic-co-speech-gesture-synthesis-and-real-ti.md)

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2401.04747>
- 项目页：<https://jeremycjm.github.io/proj/DiffSHEG/>
- 代码：<https://github.com/JeremyCJM/DiffSHEG>
- 视频：<https://www.youtube.com/watch?v=HFaSd5do-zI>
