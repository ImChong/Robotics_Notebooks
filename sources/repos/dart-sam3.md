# mkturkcan/DART

> 来源归档

- **标题：** DART — Detect Anything in Real Time
- **类型：** repo / open-vocabulary-detection / sam3 / tensorrt
- **链接：** https://github.com/mkturkcan/DART
- **权重：** https://huggingface.co/mehmetkeremturkcan/DART
- **论文：** https://arxiv.org/abs/2603.11441
- **许可：** 见仓库 LICENSE
- **语言：** Python 3.11+
- **入库日期：** 2026-09-13
- **一句话说明：** 免重训将 SAM3 转为实时多类别开放词汇检测：共享 ViT 骨干 + 批量解码 + TensorRT FP16；含 student 蒸馏骨干与 DARTF Jetson INT8 分支。
- **开源状态：** **已开源**（`pip install -e .`、TRT 导出、单图/视频/COCO eval）
- **沉淀到 wiki：** [paper-dart-sam3-realtime](../../wiki/entities/paper-dart-sam3-realtime.md)

---

## 定位

DART 不是新训练检测器，而是 **SAM3 推理栈重组**：把 class-agnostic 骨干从「每类一次」改为「全类共享」，再叠 TRT 与可选 student 骨干，服务机器人/边缘 **开放词汇多目标** 前端。

## 可运行入口

```bash
conda create -n dartsam3 python=3.11 -y && conda activate dartsam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tensorrt   # 可选，TRT 导出/推理
pip install -e .

# 一次性：共享 enc-dec engine
python -m sam3.trt.export_enc_dec --checkpoint sam3.pt --output enc_dec.onnx --max-classes 4 --imgsz 1008
python -m sam3.trt.build_engine --onnx enc_dec.onnx --output enc_dec_fp16.engine --fp16

# ViT-H 骨干 + 检测（sam3.pt 首次自动下载）
python scripts/export_hf_backbone.py --image x.jpg --imgsz 1008
```

依赖：PyTorch 2.7+、CUDA 12.6+；TensorRT 10.9+（可选）。测试环境 README 写明 Windows 11 + RTX 4080。

## 目录要点

| 路径 | 作用 |
|------|------|
| `sam3/trt/` | backbone / enc-dec ONNX 导出与 TensorRT build |
| `scripts/export_hf_backbone.py` | ViT-H 骨干 TRT |
| `scripts/export_student_trt.py` | RepViT 等蒸馏骨干 |
| `sam3/video_pipeline.py` | 视频流水线（pipelined / compiled / split backbone） |
| `dartf/` | Jetson Orin W8A8 INT8 加速分支（DARTF） |
| `scripts/eval_coco*.py` | COCO 官方评测 |

## 对 wiki 的映射

- [paper-dart-sam3-realtime](../../wiki/entities/paper-dart-sam3-realtime.md)
- [paper-sam3](../../wiki/entities/paper-sam3.md)
- [sources/papers/dart_arxiv_2603_11441.md](../papers/dart_arxiv_2603_11441.md)
