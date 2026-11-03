# S4VY 官方代码

> 来源归档（ingest）

- **仓库：** <https://github.com/Evergreen0929/S4VY>
- **核查版本：** `8fe66d4efc21947fc3ab838567f7d48642f7d00e`，2026-11-03
- **状态：** 已公开训练和推理实现，以及[模型权重](https://huggingface.co/jdzhang0929/S4VY)。
- **入口：** `infer_segmenter.py`；语言定位 `grounding/run_pipeline_scene.py` / `grounding/run_pipeline_video.py`；视频示例 `demo/forward_video.py`、`demo/grounding_video.py`。
- **限制：** README 称评测 harness、基线适配与训练数据未随仓库发布；VGGT-Ω 需单独获取。代码为 NVIDIA Source Code License-NC，分割权重和 grounding 权重许可不同，须逐项核对。
- **关联：** [S4VY 实体](../../wiki/entities/paper-s4vy.md)、[论文归档](../papers/s4vy_arxiv_2609_36875.md)。
