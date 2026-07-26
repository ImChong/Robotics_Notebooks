# SAM 2（Segment Anything Model 2）官方仓

> 来源归档

- **标题：** SAM 2: Segment Anything in Images and Videos
- **类型：** repo
- **组织：** Meta FAIR / facebookresearch
- **链接：** <https://github.com/facebookresearch/sam2>
- **别名：** <https://github.com/facebookresearch/segment-anything-2>（301 → `sam2`）
- **论文：** <https://arxiv.org/abs/2408.00714>
- **项目页：** <https://ai.meta.com/sam2>
- **Demo：** <https://sam2.metademolab.com/>
- **许可：** Apache-2.0（模型与 demo）；SA-V 为 CC BY 4.0
- **入库日期：** 2026-07-26
- **一句话说明：** SAM 2 / 2.1 图像与视频提示分割：推理 API、checkpoint、训练/微调代码与交互 demo。
- **沉淀到 wiki：** [`wiki/entities/paper-sam2.md`](../../wiki/entities/paper-sam2.md)

---

## 开源状态

**已开源**：图像/视频推理、SAM 2.1 权重、`training/`（Hydra + 分布式）、web demo、SA-V 工具与评测脚本。

---

## 仓库入口（README / training）

| 组件 | 说明 |
|------|------|
| 安装 | `git clone …/sam2 && pip install -e .`；需 `python>=3.10`，`torch>=2.5.1`；可选 `.[notebooks]` / `.[dev]` |
| Checkpoint | `checkpoints/download_ckpts.sh`；SAM 2.1 Hiera Tiny/Small/Base+/Large |
| 图像 | `SAM2ImagePredictor` + `build_sam2`；HF：`from_pretrained("facebook/sam2-hiera-large")` |
| 视频 | `build_sam2_video_predictor`：`init_state` → `add_new_points_or_box` → `propagate_in_video` |
| 自动掩码 | 与 SAM 类似的 automatic mask generator notebook |
| 训练 | `training/train.py` + `configs/sam2.1_training/...`（示例：MOSE 微调） |
| 加速 | `vos_optimized=True`（`torch.compile`）；多目标 `SAM2VideoPredictor` |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-sam2](../../wiki/entities/paper-sam2.md) | 论文实体 |
| [paper-segment-anything](../../wiki/entities/paper-segment-anything.md) / [segment-anything](./segment-anything.md) | 静态图前代 |
| [ovo-semantic-mapping](../../wiki/entities/ovo-semantic-mapping.md) | 默认 SAM 2 mask 初始化 |
| [GO2 SAM 流水线](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md) | 四足语义建图 2D 侧 |
| [sam-3d-body](./sam-3d-body.md) | 同系 Meta 3D 人体网格（不同任务） |
