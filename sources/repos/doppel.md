# Doppel 官方训练推理与标定源码

- **类型：** repo / video world model
- **代码：** https://github.com/DoppelWM/Doppel
- **论文：** https://arxiv.org/abs/2611.24925
- **项目：** https://doppelwm.github.io/
- **权重：** https://huggingface.co/sourlreapwr/doppel-emb8
- **相机参数：** https://huggingface.co/datasets/sourlreapwr/doppel-camera-extrinsics
- **核查日期：** 2026-11-20
- **核对范围：** README 全文及 doppel/inference.py 的 import、build_pipeline、from_pretrained 入口；未完成真实 GPU 推理
- **通读：** 已完成

## 原文记录

代码可训练、推理、评测和标定；`python -m doppel.inference --video clip.mp4 --render render.mp4 --actions actions.npz --robot_type panda --out out/pred.mp4`。入口通过 `DoppelWorldModel.from_pretrained` 加载 Wan2.1 I2V、VAE、CLIP 和 render-conditioned transformer，并使用 `predict` 输出 17 帧视频。作者设置 bf16、80 GB GPU、50 Euler steps、分辨率 240x432。README 另有 `python -m doppel.rollout` 与 `doppel.train`，`calibration/` 可估计缺失外参。**LICENSE MIT 仅代码**；权重 CC BY-NC 4.0，基础模型另受许可证约束。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/paper-doppel.md)
