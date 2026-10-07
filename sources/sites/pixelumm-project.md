# PixelUMM 官方项目页

- **URL：** <https://nv-tlabs.github.io/PixelUMM/>
- **论文：** <https://arxiv.org/abs/2609.38597>；[PDF](https://arxiv.org/pdf/2609.38597)；[HTML](https://arxiv.org/html/2609.38597)
- **官方源码：** <https://github.com/nv-tlabs/PixelUMM> — [仓库核查](../repos/nv-tlabs-pixelumm.md)
- **模型权重：** <https://huggingface.co/nvidia/PixelUMM>
- **作者与机构：** Cong Wei、Xuanchi Ren、Bryan Chu、Weiming Ren、Huan Ling、Jiahui Huang、Laura Leal-Taixé、Sanja Fidler、Wenhu Chen、Zian Wang、Jay Zhangjie Wu；NVIDIA、University of Waterloo
- **入库日期：** 2026-10-07
- **项目实体：** [PixelUMM](../../wiki/entities/paper-pixelumm.md)
- **论文摘录：** [arXiv 来源归档](../papers/pixelumm_arxiv_2609_38597.md)

## 项目页核查

PixelUMM 项目页链接到 arXiv、源码仓库及 Hugging Face 模型权重。官网摘要描述其直接对 RGB 像素建模：图像采用 16×16 patch，视频采用 4 帧 tubelet；Qwen3-8B 主干的理解与生成专家共享多模态 self-attention。

官网列出 F1–F8 消融，涉及 patch/tubelet 压缩率、生成输出头、模型规模和视频理解采样。线性输出头在平滑区域可能产生与 patch 对齐的明暗边界；卷积头减轻伪影但计算更高。发布 checkpoint 与公开 benchmark 仍用默认线性头。横向比较受不同训练数据影响。

## 开放状态（2026-10-07）

- **代码：** 官方 GitHub 仓库公开推理代码、数据预处理、评测入口及四任务玩具训练示例；主要源码许可为 Apache-2.0，但需遵守逐文件第三方许可声明。
- **权重：** Hugging Face 发布 PixelUMM checkpoint；模型权重与源码分属不同许可，使用 NVIDIA One-Way Noncommercial License，仅限非商业研究或评估。
- **训练复现：** 四步 toy training 不是完整训练 recipe，也不是带 optimizer state 的断点续训方案。官方估算示例训练需约 7 张、每张至少 48 GiB 显存的 GPU，输出 checkpoint 约 61 GB；玩具数据与权重需另行下载。
- **机器人验证：** 论文与项目评测为通用图像/视频任务，未给出机器人闭环控制、实机部署或动作策略评测；因此不将其归类为已验证的机器人控制方法。

## 延伸材料

- 官方代码仓库：[nv-tlabs/PixelUMM](https://github.com/nv-tlabs/PixelUMM)
- 模型卡：[nvidia/PixelUMM](https://huggingface.co/nvidia/PixelUMM)
- 论文摘录：[PixelUMM arXiv 资料](../papers/pixelumm_arxiv_2609_38597.md)
