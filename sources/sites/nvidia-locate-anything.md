# NVIDIA LPR — LocateAnything 项目页

> 来源归档

- **标题：** LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
- **类型：** site（NVIDIA Research LPR 项目页）
- **URL：** <https://research.nvidia.com/labs/lpr/locate-anything/>
- **PDF：** <https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf>
- **入库日期：** 2026-09-10
- **一句话说明：** NVIDIA LPR 发布的 LocateAnything 技术报告页：PBD 方法、Hybrid 推理、LocateAnything-Data 与多基准 SOTA 叙事。

## 开源核查（步骤 2.5，2026-09-10）

| 链接 | 状态 |
|------|------|
| [GitHub — NVlabs/Eagle/Embodied](https://github.com/NVlabs/Eagle/tree/main/Embodied) | **已开源**（代码 Apache 2.0） |
| [HF Model — nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) | **已发布**（NVIDIA License，非商业） |
| [HF Demo](https://huggingface.co/spaces/nvidia/LocateAnything) | 在线演示 |
| [HF Dataset — NVEagle/LocateAnything-Data](https://huggingface.co/datasets/NVEagle/LocateAnything-Data) | **已发布**（~2.41 TB） |

## 页面要点（策展）

- **PBD vs 量化坐标/文本数字解码：** 后者逐 token 串行；PBD 单步输出整框。
- **任务统一：** 文档理解、GUI grounding、稠密检测、OCR、指代表达、点定位。
- **产品关联：** Eagle VLM 家族；技术贡献并入 Nemotron / Cosmos 的 Computer Use 与 Visual Grounding 能力（模型卡叙述）。

## 对 wiki 的映射

- [paper-locateanything](../../wiki/entities/paper-locateanything.md)
- [locateanything 论文摘录](../papers/locateanything_arxiv_2605_27365.md)
- [Eagle Embodied 代码归档](../repos/eagle_embodied_locateanything.md)
