# ABot-World（amap-cvlab/ABot-World）

> 来源归档

- **标题：** ABot-World
- **类型：** repo
- **来源：** 阿里巴巴高德 AMAP CV Lab（ABot-World Team）
- **链接：** <https://github.com/amap-cvlab/ABot-World>
- **项目页：** <https://amap-cvlab.github.io/ABot-World/> — 见 [`sources/sites/abot-world.md`](../sites/abot-world.md)
- **Studio：** <https://abot-world.amap.com>
- **论文：** <https://arxiv.org/abs/2607.19191>
- **权重：** <https://huggingface.co/acvlab/ABot-World-0-5B-LF>（ModelScope 镜像：`amap_cvlab/ABot-World-0-5B-LF`）
- **许可：** Apache-2.0
- **基座：** Wan2.2-TI2V-5B（HF tag：`base_model:Wan-AI/Wan2.2-TI2V-5B`）
- **入库日期：** 2026-07-26
- **一句话说明：** ABot-World-0 官方开源仓：因果学生推理、低比特量化、本地 Gradio 与键盘 HUD；面向单卡 RTX 5090 级桌面实时交互世界 rollout。
- **沉淀到 wiki：** [`wiki/entities/paper-abot-world-0.md`](../../wiki/entities/paper-abot-world-0.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 待发布 |
|--------|--------|
| 推理代码、`web_client` Gradio、量化内核、配置 YAML | 双向教师权重 |
| 因果学生 `ABot-World-0-5B-LF` | 约 500 h 带动作标注训练集（已公告计划开源） |

测试环境（README）：Ubuntu 22.04、CUDA 13.3、NVIDIA RTX 5090。

---

## 仓库入口（README / 目录）

| 组件 | 说明 |
|------|------|
| 安装 | `conda create -n aworld python=3.12`；`pip install -r requirements.txt` |
| 权重下载 | `hf download acvlab/ABot-World-0-5B-LF --local-dir ./checkpoints/ABot-World-0-5B-LF`（或 ModelScope 等价命令） |
| 配置 | `configs/long_forcing_dmd.yaml`、`configs/default_config.yaml` |
| CLI 推理 | `scripts/inference.py`（动作 JSON → chunk 流式生成） |
| 因果管线 | `pipeline/causal_inference.py`（`CausalInferencePipeline`） |
| Gradio | `bash web_client/run.sh`（可选 `CUDA_ID=0`） |
| 量化 | `quantizer/`（FP8 / MXFP 等 PTQ 与 kernel） |
| Wan 模块 | `wan/`（含 Helios 系 Triton RoPE / norm kernel 致谢） |

Checkpoint 目录需含：`Wan2.2_VAE.pth`、`taew2_2.pth`、`models_t5_umt5-xxl-enc-bf16.pth`、`diffusion_pytorch_model.safetensors`、`google/umt5-xxl/`。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-abot-world-0](../../wiki/entities/paper-abot-world-0.md) | 论文实体、LongForcing 与部署包络 |
| [paper-wan-video](../../wiki/entities/paper-wan-video.md) | 上游 Wan2.2 视频先验 |
| [paper-abot-m05](../../wiki/entities/paper-abot-m05-mobile-manipulation-wam.md) | 同机构 ABot 家族，但为移动操作 WAM，非本仓 |
| [generative-world-models](../../wiki/methods/generative-world-models.md) | 生成式世界模型方法总览 |
