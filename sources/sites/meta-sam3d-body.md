# Meta SAM 3D Body（项目页与公开资产）

> 来源归档

- **标题：** SAM 3D Body — Robust Full-Body Human Mesh Recovery
- **类型：** site
- **组织：** Meta Superintelligence Labs
- **项目页（SAM 3D 产品线）：** <https://ai.meta.com/sam3d/>
- **论文项目页：** <https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/>
- **论文：** <https://arxiv.org/abs/2602.15989>
- **代码：** <https://github.com/facebookresearch/sam-3d-body>
- **权重：** <https://huggingface.co/facebook/sam-3d-body-dinov3>、<https://huggingface.co/facebook/sam-3d-body-vith>
- **数据集：** <https://huggingface.co/datasets/facebook/sam-3d-body-dataset>
- **入库日期：** 2026-09-28（链接再核；论文与仓已于 2026-05-30 入库）
- **一句话说明：** Meta SAM 3D 人体支路官方入口：单图可提示 MHR 全身 HMR，开放 checkpoint、HF 推理与门控训练数据集。
- **开源状态（步骤 2.5）：** **已开源**（PyTorch 推理仓 + 门控 HF 权重/数据；许可 SAM License）。

---

## 页面核查要点

| 资产 | 状态 |
|------|------|
| GitHub `facebookresearch/sam-3d-body` | 推理 demo、`INSTALL.md`、HF 下载脚本 |
| HF DINOv3-H+ / ViT-H checkpoint | 需 HF 访问申请 |
| `sam-3d-body-dataset` | 门控；parquet 标注需自备 COCO/MPII 等原图 |
| 在线 playground | 项目页链到 Meta AI 体验（与 `--detector_name sam3` 对齐） |

## 对 wiki 的映射

- [SAM 3D Body 实体页](../../wiki/entities/sam-3d-body.md)
- [InstantHMR](../../wiki/entities/instanthmr.md) — 基于本数据集 GT 与 MHR 输出的轻量 ONNX 学生模型
