# OmniCompass 项目页

- **标题：** OmniCompass: A Holistic Benchmark for Omni-Modal Video Grounding and Understanding
- **类型：** site / project-page
- **URL：** <https://omnicompass.netlify.app/>
- **入库日期：** 2026-10-25
- **论文：** [arXiv:2610.18829](https://arxiv.org/abs/2610.18829)
- **代码：** [PolyU-VCLab/OmniCompass](https://github.com/PolyU-VCLab/OmniCompass)
- **数据集：** [Hugging Face](https://huggingface.co/datasets/VCLab-PolyU/OmniCompass)；[ModelScope 镜像](https://modelscope.cn/datasets/liuxuyang16/OmniCompass)

## 项目页核查

本次抓取环境无法直接载入 Netlify 项目页；项目页 URL 及其资源链接由作者官方 [GitHub README](https://github.com/PolyU-VCLab/OmniCompass#readme) 对照核实。README 同时列出论文、代码、Hugging Face、ModelScope 和项目页入口。

## 一句话摘要

OmniCompass 将视频理解分成 OmniCompass-VL（视觉语言）与 OmniCompass-AV（原生音视频）两条轨道，并在时序定位、空间定位、问答任务中比较文本与文本+图像目标规格。

## 公开信息要点

- **规模：** OmniCompass-VL 为 820 个视频 / 2,460 个实例；OmniCompass-AV 为含原生音频的 240 个视频 / 720 个实例。
- **评测：** 官方 README 报告论文评估了 50 个多模态大模型。
- **开源状态：** 官方代码仓库和 Hugging Face 数据集均公开；数据卡标注 CC BY-NC-SA 4.0。AV 子集当前可下载范围以 HF / ModelScope 文件列表为准。
- **边界：** 这是模型视频 grounding / understanding 基准，不直接评机器人动作策略、控制或任务成功率。

## 关联资料

- 论文：[omnicompass_arxiv_2610_18829](../papers/omnicompass_arxiv_2610_18829.md)
- 仓库：[PolyU-VCLab/OmniCompass](../repos/polyu-vclab-omnicompass.md)
- 数据集：[VCLab-PolyU/OmniCompass](../datasets/vclab-polyu-omnicompass.md)
- wiki 实体：[OmniCompass](../../wiki/entities/paper-omnicompass.md)