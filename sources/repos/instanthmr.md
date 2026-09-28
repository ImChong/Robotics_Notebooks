# InstantHMR（轻量 ONNX MHR 回归器）

> 来源归档

- **标题：** InstantHMR
- **类型：** repo
- **组织：** mohamdev（社区工程实现）
- **代码：** <https://github.com/mohamdev/InstantHMR>
- **权重：** Hugging Face <https://huggingface.co/momolesang/InstantHMR>（`instanthmr.onnx`）
- **训练数据 / 教师（可选）：** [`facebook/sam-3d-body-dataset`](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)、[`facebook/sam-3d-body-dinov3`](https://huggingface.co/facebook/sam-3d-body-dinov3)
- **入库日期：** 2026-09-28
- **一句话说明：** RepViT-M1.5 + 9-token cross-attention decoder + CLIFF 相机条件；在 SAM 3D Body 官方 MHR 标注上训练，单文件 ONNX 约 5 ms/帧（4070 fp16）；demo 用 RF-DETR 检测，可选 pymomentum 解码稠密 MHR 网格。
- **开源状态（步骤 2.5）：** **已开源** Apache-2.0；无独立项目页，入口为 GitHub README + Hugging Face 权重。
- **沉淀到 wiki：** [InstantHMR](../../wiki/entities/instanthmr.md)

---

## 仓库要点（README ingest 快照，2026-09-28）

| 项 | 说明 |
|----|------|
| 输入 | `image (N,3,224,224)`、`cliff_cond (N,3)` |
| 输出 | `mhr_params (204)`、`shape_params (45)`、`cam_trans (3)`、70×2D/3D 关键点 |
| 检测 | RF-DETR（medium 默认）；`--detector-stride` 降频检测 |
| 网格 | 可选 Meta [MHR](https://github.com/facebookresearch/MHR) + `pymomentum`（Python ≥3.12） |
| 安装 | `python install.py`（Blackwell 自动 cu128 torch） |
| 训练 | 默认 **GT 标注**（非蒸馏）；`parquet_to_npz.py` + notebook；蒸馏见 `annotate_dataset.py` |

## 对 wiki 的映射

- 主实体页：**`wiki/entities/instanthmr.md`**
- 上游基础模型：**`wiki/entities/sam-3d-body.md`**
- 对照运行时：**`wiki/entities/sam3dbody-cpp.md`**
