# SAM 3D Body（Meta 官方推理与数据集）

> 来源归档

- **标题：** SAM 3D Body — Robust Full-Body Human Mesh Recovery
- **类型：** repo
- **组织：** Meta Superintelligence Labs / facebookresearch
- **代码：** <https://github.com/facebookresearch/sam-3d-body>
- **论文：** <https://arxiv.org/abs/2602.15989>
- **相关：** [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)、[MHR](https://github.com/facebookresearch/MHR)
- **权重：** Hugging Face `facebook/sam-3d-body-dinov3`、`facebook/sam-3d-body-vith`
- **数据集：** Hugging Face `facebook/sam-3d-body-dataset`
- **入库日期：** 2026-05-30
- **一句话说明：** 官方 PyTorch 推理、demo、checkpoint 申请与 notebook；单图（可选 keypoint/mask 提示）输出 MHR 全身网格，默认 ViTDet 或 SAM3 人体检测 + MoGe2 FOV。
- **沉淀到 wiki：** [SAM 3D Body](../../wiki/entities/sam-3d-body.md)

---

## 仓库要点（README ingest 快照）

| 项 | 说明 |
|----|------|
| 模型 | 3DB：encoder–decoder；MHR 参数（身体/脚/手） |
| 提示 | 2D keypoints、masks 等，类 SAM 交互 |
| 检测器 | 默认 ViTDet；`--detector_name sam3` 对齐在线 playground |
| 安装 | `INSTALL.md`：Python 环境 + Hugging Face checkpoint 访问申请 |
| 许可 | SAM License（见仓内 `LICENSE`） |

## 快速推理路径

```bash
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
python demo.py --image_folder <images> --output_folder <out> \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

## 对 wiki 的映射

- 主实体页：**`wiki/entities/sam-3d-body.md`**
- 论文摘录：**`sources/papers/sam_3d_body_arxiv_2602_15989.md`**
- C++ 运行时对照：**`wiki/entities/sam3dbody-cpp.md`**（[SAM3DBody-cpp](https://github.com/AmmarkoV/SAM3DBody-cpp)）
