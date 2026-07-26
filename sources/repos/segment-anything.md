# Segment Anything（SAM）官方推理仓

> 来源归档

- **标题：** Segment Anything
- **类型：** repo
- **组织：** Meta AI Research / facebookresearch
- **链接：** <https://github.com/facebookresearch/segment-anything>
- **论文：** <https://arxiv.org/abs/2304.02643>
- **项目页：** <https://segment-anything.com/>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-26
- **一句话说明：** 运行 SAM 推理、下载 ViT-B/L/H checkpoint、提示式与自动掩码 notebook，以及 mask decoder 的 ONNX 导出。
- **沉淀到 wiki：** [`wiki/entities/paper-segment-anything.md`](../../wiki/entities/paper-segment-anything.md)

---

## 开源状态

**已开源**（推理 / 权重 / notebook / ONNX / 简易 web demo）。README 顶部指向视频继任者 [SAM 2](https://github.com/facebookresearch/sam2)（旧名 `segment-anything-2` 会 301 到 `sam2`）。

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | `pip install git+https://github.com/facebookresearch/segment-anything.git` 或本地 `pip install -e .` |
| 依赖 | `python>=3.8`，`pytorch>=1.7`，`torchvision>=0.8`；可选 opencv / pycocotools / onnx |
| 提示推理 | `SamPredictor`：`set_image` → `predict(points/boxes/masks)` |
| 全图自动掩码 | `SamAutomaticMaskGenerator`；CLI：`scripts/amg.py` |
| Checkpoint | `vit_h` / `vit_l` / `vit_b`（`dl.fbaipublicfiles.com/segment_anything/...`） |
| ONNX | `scripts/export_onnx_model.py`；浏览器 demo 见 `demo/` |
| Notebook | `notebooks/predictor_example.ipynb`、`automatic_mask_generator_example.ipynb` |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-segment-anything](../../wiki/entities/paper-segment-anything.md) | 论文实体 |
| [paper-sam2](../../wiki/entities/paper-sam2.md) / [sam2](./sam2.md) | 图像+视频统一继任 |
| [ovo-semantic-mapping](../../wiki/entities/ovo-semantic-mapping.md) | 下游可用 SAM1 初始化 mask |
| [GO2 SAM 流水线](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md) | 机器人 2D→3D 语义选型 |
