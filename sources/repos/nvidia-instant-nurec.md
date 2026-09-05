# NVIDIA/instant-nurec

> 来源归档（ingest）

- **标题：** InstantNuRec（官方推理 CLI）
- **类型：** repo
- **组织：** NVIDIA
- **代码：** <https://github.com/NVIDIA/instant-nurec>
- **Stars / Forks：** ~198 / 0（2026-09-05 核查）
- **许可证：** Apache-2.0（`LICENSE.txt`）；第三方见 `THIRD_PARTY_LICENSE.txt`
- **默认分支：** `main`
- **语言：** Python 3.11（`uv` + `setup.sh`）
- **论文：** arXiv:2607.14203 — 归档见 [sources/papers/instant_nurec_arxiv_2607_14203.md](../papers/instant_nurec_arxiv_2607_14203.md)
- **项目页：** <https://research.nvidia.com/labs/sil/projects/instant-nurec/> — 归档见 [sources/sites/nvidia-research-instant-nurec.md](../sites/nvidia-research-instant-nurec.md)
- **产品文档：** <https://docs.nvidia.com/nurec/> — 归档见 [sources/sites/nvidia-nurec-docs.md](../sites/nvidia-nurec-docs.md)
- **权重：** Hugging Face `nvidia/instant-nurec`
- **演示数据：** Hugging Face `nvidia/PhysicalAI-Autonomous-Vehicles-NCore`（门控，需接受条款 + `hf auth login`）
- **入库日期：** 2026-09-05
- **一句话说明：** 官方 **部分开源**：原生 Python 前向把 **NCore V4** 片段打成 **静态 3DGS PLY + 天空 sidecar**；论文里的动态层 / 完整研究模型 **不在本导出路径**。输出可初始化 Docker 版 [NuRec](https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html) 逐场景精修。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-instant-nurec.md`](../../wiki/entities/paper-instant-nurec.md)、[`wiki/entities/nvidia-nurec.md`](../../wiki/entities/nvidia-nurec.md)

## 开源边界（步骤 2.5，2026-09-05）

项目页、文档与 README 均链到本仓与 HF 权重。以 **页面实际链接** 为准，勿只读 PDF「code is available」。

| 状态 | 内容 |
|------|------|
| **已发布** | `run_inference.py` / `run.sh`：NCore ingest → chunk → eager PyTorch 前向 → 静态 PLY + `.sky.npz` / `.sky.png`；`--merge` 视锥所有权合并 + KL 体素化（默认目标 2M 高斯）；`--render-preview` / `--render-video`（需 `uv sync --extra render` + ffmpeg）；三套权重 `pa-front` / `pa-multiview` / `pq-front` 首次运行自动下载 |
| **未随本仓** | README 明确：独立 CLI **只导出静态高斯 + 观测天空 sidecar**；论文动态层、完整分层世界、训练代码 **不在本静态导出路径** |
| **产品侧（非本仓）** | NuRec 逐场景训练 / USDZ 在 NGC 容器 `nvcr.io/nvidia/nre/nre-ga:26.04`（需 `NGC_API_KEY`），不是 Apache 推理仓 |
| **数据** | 公开演示 clip ~2 GB，但数据集 **门控** |

## 仓库结构（顶层，对齐 README）

```
instant-nurec/
├── instant_nurec/
│   ├── cli.py
│   ├── pretrained.py              # 首次从 HF 拉权重
│   ├── datasets/                  # ncorev4 ingest
│   ├── model/                     # backbone / DPT / sky / static_core
│   └── predict/                   # PLY / sky 导出 + merge
├── run_inference.py
├── run.sh
├── setup.sh
└── tests/
```

## 可运行入口（对齐 README）

```bash
git clone https://github.com/NVIDIA/instant-nurec.git
cd instant-nurec
./setup.sh
source .venv/bin/activate

# 门控数据集：先在 HF 接受条款并 hf auth login
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NCore --repo-type dataset \
  --include "clips/000da9de-0ee5-465a-9a2d-e7e91d3016bb/*" \
  --local-dir ./demo_clip

python run_inference.py \
  --model pa-front \
  --ncore-path ./demo_clip/clips/000da9de-0ee5-465a-9a2d-e7e91d3016bb/pai_000da9de-0ee5-465a-9a2d-e7e91d3016bb.json \
  --output-dir ./demo_output \
  --merge
```

成功：`--merge` 后一条 PLY，约 **1.88M** 高斯（从约 2.87M 合并体再体素化到默认 2M 目标）。无 Docker。硬件下限跟 [NuRec Hardware](https://docs.nvidia.com/nurec/basics/hardware.html)。

| `--model` | 说明 | 默认输入 |
|-----------|------|----------|
| `pa-front` | 前视，**Dense** 像素对齐高斯（默认） | 18 × `camera_front_wide_120fov`，784×448 |
| `pa-multiview` | 1/3/5 相机 Dense | 每相机 18 帧，504×280 |
| `pq-front` | 前视 **Selective** 点查询（更少高斯） | 同 `pa-front` 分辨率 |

本地权重：`export INSTANT_NUREC_FULL_PT=/path/to/*.pth`。相机内参差太多要先做 [docs/camera_rectification.md](https://github.com/NVIDIA/instant-nurec/blob/main/docs/camera_rectification.md) 一次整流。

## 对 wiki 的映射

- [Instant NuRec 论文实体](../../wiki/entities/paper-instant-nurec.md)
- [NVIDIA Omniverse NuRec](../../wiki/entities/nvidia-nurec.md)
- [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)
- [Sim2Real](../../wiki/concepts/sim2real.md)
