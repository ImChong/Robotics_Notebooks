# SAM 3.1 Release Notes（2026-03-27）

> 来源归档（ingest）

- **标题：** SAM 3.1 — Object Multiplex 与推理优化
- **类型：** release-notes / foundation-model / video-segmentation
- **组织：** Meta / facebookresearch
- **原文：** <https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md>
- **Blog：** <https://ai.meta.com/blog/segment-anything-model-3/>
- **代码：** <https://github.com/facebookresearch/sam3>（同 SAM 3 仓；见 [`sources/repos/sam3.md`](../repos/sam3.md)）
- **权重：** <https://huggingface.co/facebook/sam3.1>
- **论文（架构细节 Appendix H）：** <https://arxiv.org/abs/2511.16719>
- **入库日期：** 2026-09-29
- **一句话说明：** SAM 3 视频管线的 **Object Multiplex** 共享内存联合多目标跟踪 + 推理编译/批处理优化；新 checkpoint 与 `sam3.1_video_predictor_example.ipynb`。

## 开源状态（步骤 2.5）

- **仓库核查（2026-09-29）：** 同仓 `facebookresearch/sam3`；Release Notes、`examples/sam3.1_video_predictor_example.ipynb` 与 HF `facebook/sam3.1` 权重可公开获取。
- **结论：** **已开源**（在 SAM 3 基线上增量发布 checkpoint 与示例，非新论文仓）。

## 摘录 1：Object Multiplex

- SAM 3 视频侧对每个跟踪目标 **独立** 跑管线 → 目标数线性放大算力。
- **Object Multiplex：** 将目标分入固定容量 bucket，**联合** 处理，削减重复计算；技术细节见论文 **Appendix H**。
- **单 H100、128 目标：** 相对 2025-11 SAM 3 发布约 **~7×** 加速（Release Notes 叙述）。

**对 wiki 的映射：** 多实例视频 PCS / 机载多目标跟踪选型时优先 **SAM 3.1 + Object Multiplex**；见 [`wiki/entities/paper-sam3.md`](../../wiki/entities/paper-sam3.md)。

## 摘录 2：推理与工程优化

- 检测–跟踪关联等启发式中 **减少 CPU–GPU 同步**。
- 增强 **`torch.compile`** 与算子融合。
- **批处理后处理** 与 **批处理视觉编码器** 提高 GPU 利用率。

**对 wiki 的映射：** 部署侧与 [DART](../../wiki/entities/paper-dart-sam3-realtime.md) 等 TRT 路线互补——官方路径先吃 compile + multiplex 再考虑导出。

## 摘录 3：基准（Release Notes 表）

**Video PCS（文本提示，节选）：**

| 模型 | YT-Temporal-1B cgF1 | SA-V cgF1 | MOSEv2 J&F（VOS） |
|------|---------------------|-----------|-------------------|
| SAM 3 | 50.8 | 30.3 | 60.3 |
| SAM 3.1 | 52.9 (+2.1) | 30.5 | 62.3 (+2.0) |

- SA-Co/VEval 上 **混合结果**（部分 split 略降、部分升）；VOS **7 项中 6 项** 改善。

**对 wiki 的映射：** 图像 PCS 能力仍以 SAM 3 论文为准；**3.1 主要换视频 多目标效率与 VOS/部分 video PCS**。

## 建议 wiki 动作

- 升级 [`wiki/entities/paper-sam3.md`](../../wiki/entities/paper-sam3.md)（同 arXiv:2511.16719）：增 SAM 3.1 小节、HF 权重、运行时序图 Object Multiplex 分支。
- 更新 [`sources/repos/sam3.md`](../repos/sam3.md) 与新建 [`sources/sites/meta-sam3.md`](../sites/meta-sam3.md)。
