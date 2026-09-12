# jdk9405/Wid3R

> 来源归档

- **标题：** Wid3R — Wide Field-of-View 3D Reconstruction via Camera Model Conditioning
- **类型：** repo
- **组织：** jdk9405（Dongki Jung 等，UMD × NAVER LABS）
- **代码：** <https://github.com/jdk9405/Wid3R>
- **论文：** <https://arxiv.org/abs/2602.05321>
- **项目页：** <https://jdk9405.github.io/Wid3R/>
- **权重：** Google Drive [`wid3r.bin`](https://drive.google.com/file/d/1N6nneTNLg-On_TRR1Yt2EwAEKtBd6sVf/view?usp=sharing)
- **会议：** ECCV 2026
- **入库日期：** 2026-09-12
- **一句话说明：** 宽 FoV 前馈多视图重建；`demo_gradio.py` 交互推理 + `scripts/train_wid3r.py` 训练；Pi3 初始化；评测在 `evaluation/`。
- **沉淀到 wiki：** [`wiki/entities/paper-wid3r.md`](../../wiki/entities/paper-wid3r.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源** |
| **推理** | `demo_gradio.py`（fisheye / 360 相机模式；GLB 导出） |
| **训练** | `scripts/train_wid3r.py` + Hydra `configs/`；Pi3 `model.safetensors` 初始化 |
| **评测** | `evaluation/mv_recon/`、`monodepth/`、`relpose/` |
| **待发布** | README TODO：**数据预处理代码** |
| **环境** | Conda `environments.yml`（Python 3.10, PyTorch 2.3.1, CUDA 12.1, Gradio 5.45） |

## README 要点（2026-09-12）

### 安装与 GPU

```bash
git clone https://github.com/jdk9405/Wid3R.git
cd Wid3R
conda env create -f environments.yml
conda activate wid3r
```

### 推理（Gradio）

1. 下载 `pretrained_weights/wid3r.bin`
2. 设置 `demo_gradio.py` 中 `CKPT_DIR`
3. `python demo_gradio.py` → 上传有序重叠图像或选 `360 Example` → 选相机模型 → Reconstruct

### 训练

```bash
wget https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors \
  -O pretrained_weights/model.safetensors

accelerate launch --num_processes 1 \
  --config_file configs/accelerate/ddp.yaml \
  scripts/train_wid3r.py train=train_wid3r \
  model.ckpt=pretrained_weights/model.safetensors
```

### 目录结构（运行时）

| 路径 | 用途 |
|------|------|
| `wid3r/` | 模型实现 |
| `cam_utils/` | 相机模型与射线工具 |
| `configs/` | Hydra 数据/模型/训练配置 |
| `scripts/train_wid3r.py` | 训练入口 |
| `demo_gradio.py` | Gradio 推理与可视化 |
| `evaluation/` | 多视图重建 / 单目深度 / 相对位姿评测 |

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-wid3r.md`](../../wiki/entities/paper-wid3r.md)
- 项目页：[`sources/sites/wid3r-site.md`](../sites/wid3r-site.md)
- 论文摘录：[`sources/papers/wid3r_arxiv_2602_05321.md`](../papers/wid3r_arxiv_2602_05321.md)
