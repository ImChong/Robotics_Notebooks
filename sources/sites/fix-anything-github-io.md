# FixAnything 项目页（ECCV 2026）

> 来源归档（ingest）

- **标题：** FixAnything — 3D-Consistent Rendering Refinement via Video Generative Priors
- **类型：** site（官方项目页）
- **发布方：** Khiem Vuong / Deva Ramanan* / Srinivasa Narasimhan*（Carnegie Mellon University）
- **原始链接：** <https://fix-anything.github.io/>
- **论文：** <https://arxiv.org/abs/2608.23549>
- **代码：** <https://github.com/kvuong2711/fix-anything>
- **权重：** <https://huggingface.co/kvuong2711/fix-anything>
- **入库日期：** 2026-09-01
- **一句话说明：** ECCV 2026 落地页：四类 3D 表示对比 slider、SFT + Flow-DPO 方法卡、DL3DV / MipNeRF-360 / LLFF 结果与不确定性分析。

## 项目页 / 源码开放核查（步骤 2.5 · 2026-09-01）

| 核查项 | 结论 |
|--------|------|
| **项目页 Code 链接** | 明确链到 GitHub `kvuong2711/fix-anything` |
| **权重** | Hugging Face `kvuong2711/fix-anything`（FixAnything LoRA） |
| **开放程度** | **已开源（推理）**：`run_inference.py`、`run_mapanything.py`、`download_models.py`；Apache-2.0；README 写明「This repository contains the inference code」 |
| **训练代码** | **未发布**：SFT / Flow-DPO 训练脚本不在仓内 |
| **依赖基座** | Wan2.1-I2V-14B-480P（~60 GB，脚本自动下载） |

## 摘录要点（与论文分工）

- **对外叙事：** 退化渲染仍保留相机轨迹与粗场景布局 → 视频先验做清理；稀疏点云即可作相机控制信号。
- **方法卡：** Stage I LoRA SFT（mask 通道拼接）；Stage II COLMAP 位姿 AUC 奖励 + Flow-DPO。
- **演示：** 3DGS / NeRF / Mesh / Sparse points 四表示 slider；DL3DV-Drone 长轨迹分块；MipNeRF-360 / LLFF 跨数据集。

## 对 wiki 的映射

- [FixAnything（论文）](../../wiki/entities/paper-fixanything.md)
- 姊妹归档：[论文摘录](../papers/fixanything_arxiv_2608_23549.md)、[代码仓](../repos/fix-anything.md)
