# nv-dvl/vgg-ttt

> 来源归档

- **标题：** VGG-T³ — Offline Feed-Forward 3D Reconstruction at Scale
- **类型：** repo
- **组织：** NVIDIA DVL（nv-dvl）
- **代码：** <https://github.com/nv-dvl/vgg-ttt>
- **论文：** <https://arxiv.org/abs/2602.23361>
- **项目页：** <https://research.nvidia.com/labs/dvl/projects/vgg-ttt/>
- **权重：** <https://huggingface.co/nvidia/vgg-ttt>
- **Stars：** ~195（2026-09-09）
- **入库日期：** 2026-09-09
- **一句话说明：** CVPR 2026 官方实现：VGGT 兼容 API 的 **TTT 线性全局 attention** 3D 重建；含 `infer` 推理、`demo.py` 交互可视化、`evaluation/` 复现脚本与 **部分** 训练 harness。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-vgg-ttt.md`](../../wiki/entities/paper-vgg-ttt.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（推理 + 评测 + demo）；训练 **部分开源**（harness 有、数据集实现缺） |
| **权重** | **已发布** — Hugging Face [nvidia/vgg-ttt](https://huggingface.co/nvidia/vgg-ttt)（NVIDIA OneWay Noncommercial） |
| **许可** | 仓库主体 NVIDIA OneWay Noncommercial；子目录见 README（VGGT / LaCT / CUT3R 第三方许可） |
| **商业** | README / HF 卡：**非商业科研与教育**；机器人量产部署需另审许可 |

## README 要点（2026-09-09）

### 环境与安装

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install .
# 评测额外：pip install .[evaluation]
```

### 核心入口

| 路径 | 用途 |
|------|------|
| `vggttt/nets/vggt/models/vggt.py` | `VGGT.from_pretrained("nvidia/vgg-ttt")` + `infer()` |
| `vggttt/nets/vggt/img.py` | `load_and_preprocess_images` |
| `vggttt/nets/ttt.py` / `ttt_attention.py` | TTT 线性全局 attention 实现 |
| `vggttt/demo.py` | Viser + Gradio 交互 3D 重建 |
| `vggttt/evaluation/pointmaps/eval.py` | 点图基准（DTU / ETH3D / NRGBD / 7-Scenes） |
| `vggttt/evaluation/visloc/eval.py` | 视觉定位（7-Scenes / Wayspots） |
| `vggttt/train.py` | 训练 harness（数据集实现 **未随仓发布**） |

### 推理输出字典

- `pose` — `[#images, 4, 4]` camera-to-world
- `intrinsics` — `[#images, 3, 3]` pinhole
- `pts3d` — `[#images, H, W, 3]` 世界坐标 per-pixel 点
- `conf` — per-pixel 置信度
- `depth` — per-pixel 深度

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-vgg-ttt.md`](../../wiki/entities/paper-vgg-ttt.md)
- 项目页：[`sources/sites/nvidia-dvl-vgg-ttt.md`](../sites/nvidia-dvl-vgg-ttt.md)
- 论文摘录：[`sources/papers/vgg_ttt_arxiv_2602_23361.md`](../papers/vgg_ttt_arxiv_2602_23361.md)
