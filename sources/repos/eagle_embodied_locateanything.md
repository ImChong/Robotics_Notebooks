# NVlabs/Eagle — Embodied（LocateAnything）

> 来源归档

- **仓库：** <https://github.com/NVlabs/Eagle>
- **子目录：** <https://github.com/NVlabs/Eagle/tree/main/Embodied>
- **类型：** repo / vlm / visual-grounding / inference / evaluation
- **维护方：** NVIDIA（NVlabs）
- **许可：** 代码 **Apache 2.0**；模型权重见 [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B)
- **入库日期：** 2026-09-10
- **一句话说明：** LocateAnything 官方实现入口：训练数据准备、评测脚本、Hybrid/Fast/Slow 推理与 HF 权重加载；隶属 Eagle VLM 仓库 Embodied 子树。

## 开源状态（步骤 2.5）

- **已开源：** README 链到 Paper / HF Model / HF Demo / 项目页；含 **Training · Data Preparation · Evaluation · Detailed Results** 章节。
- **权重：** `nvidia/LocateAnything-3B`；可选高吞吐推理工具（HF 模型卡 Updates）。
- **数据：** `NVEagle/LocateAnything-Data`；`tools/download_subset.py` 子集下载；部分上游媒体需 `hydrate_restricted_media.py`。

## 工程入口（策展）

| 组件 | 说明 |
|------|------|
| Embodied/ | LocateAnything 训练、评测、推理主目录 |
| HF `LocateAnything-3B` | 3B 权重；MoonViT + Qwen2.5-3B |
| HF Spaces Demo | 在线 grounding 演示 |
| `document/RESULTS.md` | 详细基准结果（仓库内） |

## 对 wiki 的映射

- [paper-locateanything](../../wiki/entities/paper-locateanything.md)
- [locateanything 论文摘录](../papers/locateanything_arxiv_2605_27365.md)
- [NVIDIA 项目页归档](../sites/nvidia-locate-anything.md)
