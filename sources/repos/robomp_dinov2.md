# RoboMP-DINOv2 官方仓库

> 来源归档（repo）

- **标题：** RoboMP-DINOv2: Prompts, Not Filters for Robust Robot Manipulation
- **类型：** repo
- **组织 / 作者：** han20192019（Han Qi、Heng Yang，Harvard SEAS）
- **代码：** <https://github.com/han20192019/RoboMP_DINOv2>
- **论文：** <https://arxiv.org/abs/2609.25506>
- **许可证：** MIT
- **入库日期：** 2026-10-18
- **核对范围：** 全仓通读 `_tree.txt`（共 12 文件）及源码：`README.md`、`MODEL_CARD.md`、`pyproject.toml`、`hubconf.py`、`examples/minimal_inference.py`、`scripts/convert_legacy_pfse_checkpoint.py`、`src/robomp_dinov2/{__init__.py,models.py,modeling.py,augmentations.py}`
- **通读：** 已完成
- **一句话说明：** 发布可独立嵌入下游机器人策略的 `robomp_dinov2` 视觉表征模块（`RoboMPDINOv2`、`RoboMPDINOv2MCR`、`MaskedRegionColorRandomization`）、最小前向示例与旧名 `PFSE` checkpoint 转换工具；策略网络、分割器与仿真基准未内置。

## 开源状态与开放边界（2026-11-14 源码核查）

- **结论：** **部分开源 — 仅释放视觉编码器包**

| 组件 | 仓库开放状态 | 核查证据 |
|------|--------------|----------|
| 视觉表征模块 | **已开源可运行** | `src/robomp_dinov2/models.py` / `modeling.py` 实现 `RoboMPDINOv2`、`RoboMPDINOv2MCR` |
| MCR 颜色增强 | **已开源可运行** | `src/robomp_dinov2/augmentations.py` 实现 `MaskedRegionColorRandomization`（训练期随机色替换并集槽像素，推理直通） |
| Hub & 旧权重工具 | **代码已开源；无模型权重** | `hubconf.py` 对 `pretrained=True` 明确抛出 `ValueError("No full RoboMP-DINOv2 checkpoint is bundled yet")`；`scripts/convert_legacy_pfse_checkpoint.py` 提供旧内部命名 `PFSE` 状态字典转换 |
| 策略 / 文本 / 分割 / 仿真 | **未包含** | `modeling.py` 模块说明显式标注下游 Diffusion Policy、CLIP 文本编码器、Grounding DINO + SAM 掩码管线及 7 任务仿真环境均不在本包内 |

## 仓库核心结构与配置默认值

| 文件路径 | 核心符号与默认设计 |
|----------|--------------------|
| `src/robomp_dinov2/modeling.py` | `RoboMPDINOv2Config`（`dinov2_vitb14`，`image_size=112`，`patch_size=14` → `8×8` token，`backbone_dim=768`，`hidden_dim=256`，`num_layers=3`，`num_heads=8`，`mlp_ratio=4.0`，`num_mask_slots=2`，`slot_names=("object","target")`，`randomize_slots=("object",)`）；前向链 `_resize_images` → `_normalize_images` → `_canonicalize_masks` → `_extract_patch_tokens` → `proj` + `pos_embed` + `mask_prompts` → `transformer` → mean-pool `[B, 256]` |
| `src/robomp_dinov2/models.py` | `RoboMPDINOv2`、`RoboMPDINOv2MCR`（内部集成 `MaskedRegionColorRandomization`，仅 `self.training` 时执行颜色随机化） |
| `src/robomp_dinov2/augmentations.py` | `MaskedRegionColorRandomization`：对目标槽位并集像素替换为 `[0, 1]^3` 均匀随机颜色 |
| `examples/minimal_inference.py` | 不加载完整 DINOv2 hub 权重的本地形状烟雾/最小推理演示，展示随机张量 `(B, 3, 112, 112)` 与 `(B, 2, 112, 112)` 输入流经 `RoboMPDINOv2MCR` 输出 `[B, 256]` |

## 对 wiki 的映射

- [paper-robomp-dinov2](../../wiki/entities/paper-robomp-dinov2.md)
- [论文摘录](../papers/robomp_dinov2_arxiv_2609_25506.md)
