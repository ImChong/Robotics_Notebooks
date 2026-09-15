# Go-with-the-Track（Eyeline-Labs/Go-with-the-Track）

> 来源归档

- **标题：** Go-with-the-Track
- **类型：** repo
- **组织：** Eyeline Labs / Netflix（作者归属）
- **主页：** <https://github.com/Eyeline-Labs/Go-with-the-Track>
- **项目页：** <https://eyeline-labs.github.io/Go-with-the-Track/>
- **论文：** <https://arxiv.org/abs/2606.20891>
- **HF 模型：** <https://huggingface.co/Eyeline-Labs/Go-with-the-Track>
- **HF 数据集：** <https://huggingface.co/datasets/Eyeline-Labs/Go-with-the-Track>
- **入库日期：** 2026-09-15
- **一句话说明：** SIGGRAPH 2026 官方实现：在 Wan2.2 上注入 reference-anchored point-track 条件；`run_inference_dataset.py` 批量推理；480P/720P checkpoint 与 eval 数据在 HF。
- **沉淀到 wiki：** [`wiki/entities/paper-go-with-the-track.md`](../../wiki/entities/paper-go-with-the-track.md)

---

## 开源状态

**已开源（截至 2026-09-15）**：Apache-2.0；Ubuntu 22.04 + CUDA 11.8 测试。

| 路径 | 角色 |
|------|------|
| `run_inference_dataset.py` | 主推理入口（多 GPU） |
| `inference.py` | 单样本推理逻辑 |
| `model_*.py` | embedder、adapter、downsample、Wan pipeline 封装 |
| `configs/480P.sh` / `720P.sh` | 分辨率与 GPU 配置 |
| `data_preprocess/` | 数据预处理 |
| `DiffSynth-Studio/` | 依赖子模块/集成 |

---

## 推荐复现入口

```bash
conda create -n gwtt python=3.10
conda activate gwtt
git clone https://github.com/Eyeline-Labs/Go-with-the-Track.git
cd Go-with-the-Track
pip install -e . --no-build-isolation

# 下载 checkpoint 与 eval 数据（见 README hf download）
python run_inference_dataset.py \
  --config_path "./configs/480P.sh" \
  --path_to_dataset ./eval_data \
  --output_folder ./tmp_output/wan22_480P \
  --gpus 1
```

---

## 与仓库内实体的关系

- 论文实体：[paper-go-with-the-track.md](../../wiki/entities/paper-go-with-the-track.md)
- 上游视频骨干：[wan2.1.md](./wan2.1.md) / [paper-wan-video.md](../../wiki/entities/paper-wan-video.md)
- 概念：[video-as-simulation.md](../../wiki/concepts/video-as-simulation.md)
