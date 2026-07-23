# Wan-Move: Motion-Controllable Video Generation via Latent Trajectory Guidance（arXiv:2512.08765）

> 来源归档（ingest）

- **标题：** Wan-Move: Motion-Controllable Video Generation via Latent Trajectory Guidance
- **类型：** paper / motion-controllable video / latent trajectory / I2V / MoveBench
- **arXiv：** <https://arxiv.org/abs/2512.08765>（PDF：<https://arxiv.org/pdf/2512.08765.pdf>）
- **会议：** NeurIPS 2025
- **项目页：** <https://wan-move.github.io/>
- **代码：** <https://github.com/ali-vilab/Wan-Move>
- **权重：** HF <https://huggingface.co/Ruihang/Wan-Move-14B-480P>；ModelScope <https://modelscope.cn/models/churuihang/Wan-Move-14B-480P>
- **基准：** HF <https://huggingface.co/datasets/Ruihang/MoveBench>
- **作者：** Ruihang Chu*、Yefei He*、Zhekai Chen*、Shiwei Zhang、Xiaogang Xu、Bin Xia、Dingdong Wang、Hongwei Yi、Xihui Liu、Hengshuang Zhao、Yu Liu、Yingya Zhang、Yujiu Yang 等（Tongyi Lab / 清华 / HKU / CUHK）
- **机构：** 阿里巴巴（Alibaba）通义实验室、清华大学（Tsinghua）、香港大学（HKU）、香港中文大学（CUHK）
- **入库日期：** 2026-07-23
- **一句话说明：** 在 **不改 I2V 架构** 的前提下，把稠密点轨迹投到 latent，沿轨迹复制首帧特征作运动引导；基于 **Wan-I2V-14B** 微调，生成 **5 s / 480p** 可控视频，用户研究称运动可控性可比 Kling 1.5 Pro Motion Brush；配套 **MoveBench**（~1018 clip、54 类）。

## 开源状态（项目页 + 仓库核查，2026-07-23）

- **已开源：** 项目页 / README 挂 [`ali-vilab/Wan-Move`](https://github.com/ali-vilab/Wan-Move)（**Apache-2.0**）+ HF/ModelScope 14B-480P 权重 + MoveBench 数据与 `MoveBench/bench.py` + Gradio demo。

## 摘要级要点

- **瓶颈：** 已有运动控制或粒度粗（框/掩码），或依赖额外运动编码器 / ControlNet，难与强 I2V 骨干同尺度微调。
- **核心：** 像素轨迹 → VAE 空间映射 → 沿 latent 轨迹复制首帧特征，**直接改写** \(z_{\text{image}}\)；无辅助模块。
- **训练：** ~**200 万** 720p 视频两阶段过滤；CoTracker 稠密轨迹；5% 丢运动条件以保留原 I2V 能力。
- **与机器人关系：** 本身是通用视频运动控制；[Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) 将其列为视觉基线对照，并说明 Wan 系可控视频是机器人像素条件 WM 的重要先验栈。

## 核心论文摘录（MVP）

### 1) Latent Trajectory Guidance

- **链接：** §3.2；Eq. (1)–(3)；Fig. 2
- **摘录要点：** 首帧 + zero-pad 经 VAE 得 \(z_{\text{image}}\)；轨迹按 \(f_s,f_t\) 映射到 latent；把首帧特征复制到后续帧轨迹位置。
- **对 wiki 的映射：**
  - [Wan-Move](../../wiki/entities/paper-wan-move.md) — 核心机制。
  - [Wan](../../wiki/entities/paper-wan-video.md) — 基座 I2V。

### 2) 无架构改动的可扩展微调

- **链接：** §3.3；§1
- **摘录要点：** 初始化自 Wan-I2V，flow-matching 微调；相对 Motion Prompting 等「像素随机嵌入 + ControlNet」路线，本方法保留局部纹理上下文且无需额外模块。
- **对 wiki 的映射：**
  - [Wan-Move](../../wiki/entities/paper-wan-move.md) — 工程可扩展性。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 条件注入谱系。

### 3) MoveBench

- **链接：** §4；Fig. 3–5
- **摘录要点：** **1018** 条 5 s 视频、**54** 内容类；点轨迹 + 分割掩码；人标 + SAM 混合标注；相对 DAVIS / VIPSeg / MagicBench 更长、更全。
- **对 wiki 的映射：**
  - [Wan-Move](../../wiki/entities/paper-wan-move.md) — 评测资产。

## BibTeX

```bibtex
@article{chu2025wan,
  title   = {Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance},
  author  = {Chu, Ruihang and He, Yefei and Chen, Zhekai and Zhang, Shiwei and
             Xu, Xiaogang and Xia, Bin and Wang, Dingdong and Yi, Hongwei and
             Liu, Xihui and Zhao, Hengshuang and others},
  journal = {arXiv preprint arXiv:2512.08765},
  year    = {2025}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-wan-move.md`](../../wiki/entities/paper-wan-move.md)
- 代码归档：[`sources/repos/wan-move.md`](../repos/wan-move.md)
- 项目页：[`sources/sites/wan-move-github-io.md`](../sites/wan-move-github-io.md)
- 互链：[Wan](../../wiki/entities/paper-wan-video.md)、[Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)
