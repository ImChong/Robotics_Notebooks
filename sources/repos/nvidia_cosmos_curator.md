# NVIDIA/cosmos-curator

> 来源归档

- **标题：** Cosmos Curator
- **类型：** repo
- **组织：** NVIDIA
- **代码：** <https://github.com/NVIDIA/cosmos-curator>
- **文档：** 仓内 [`docs/`](https://github.com/NVIDIA/cosmos-curator/tree/main/docs)；托管服务见 [Cosmos Curator LHA 文档](https://docs.nvidia.com/cosmos-curator-lha/current/introduction.html)
- **Stars：** ~264（2026-09-06）
- **入库日期：** 2026-09-06
- **一句话说明：** NVIDIA **Physical AI 视频策展开源框架**：Ray + GPU 流式管线做切镜、过滤、embedding、VLM 字幕与语义去重，产出 Cosmos WFM 后训练用的 WebDataset。
- **沉淀到 wiki：** 是 → [`wiki/entities/cosmos-curator.md`](../../wiki/entities/cosmos-curator.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Apache-2.0；依赖第三方模型另计许可） |
| **代码** | <https://github.com/NVIDIA/cosmos-curator> |
| **核心依赖** | [cosmos-xenna](https://github.com/nvidia-cosmos/cosmos-xenna)（git submodule；GPU 流式管线框架） |
| **托管服务** | [Cosmos Curator LHA](https://docs.nvidia.com/cosmos-curator-lha/current/introduction.html) — DGX Cloud 上 GPU 加速策展；S3 双桶或 ZIP 上传；NGC WebUI 或 API |
| **产品页** | <https://www.nvidia.com/en-us/ai/cosmos/> |
| **许可** | 源码 Apache-2.0；Cosmos 模型权重走 NVIDIA Open Model License |

## README 要点（2026-09-06）

- **定位：** 为 [Cosmos WFM](https://www.nvidia.com/en-us/ai/cosmos/) 训练数据生成提供视频处理与策展；底层流式框架已独立开源为 **Cosmos-Xenna**（Ray 多节点多 GPU）。
- **能力：** 视频切分、标注、过滤、去重、数据集生成；模块化 pipeline；本地 Docker / Slurm / NVCF / DGX Cloud 部署。
- **CLI 入口：** `cosmos-curator`（Pixi 环境）；子命令含 `image build`、`local launch`、`slurm`、`nvcf`。
- **目录：** `cosmos_curator/`（client / core / models / pipelines / scripts）、`cosmos-xenna` submodule、`packages/` Docker 镜像、`examples/` 配置模板。

## 三条参考视频管线（docs/curator/reference/video-pipelines.md）

| 管线 | 作用 |
|------|------|
| **split-annotate** | TransNetV2 切镜 → H264 转码 → 运动/美学过滤 → InternVideo2 / Cosmos-Embed1 embedding → VLM 字幕 → 写 clips + metadata |
| **dedup** | 基于 split-annotate 产出的 embedding 做语义去重 |
| **shard-dataset** | 把 clips + captions（可选 dedup 结果）打成 **WebDataset**，供 Cosmos-Predict2 Video2World 后训练 |

split-annotate 还可产出 `cosmos_predict2_video2world_dataset/`（T5 caption + mp4 帧窗）、Milvus 索引 parquet、caption 质量统计等。

## 运行门槛（End User Guide 摘录）

| 项 | 要求 |
|----|------|
| 主机内存 | ≥ 32 GB |
| 磁盘 | ≥ 200 GB |
| GPU | 算力 ≥ 8.0；hello-world ≥ 4 GB VRAM；参考视频管线 ≥ **48 GB** |
| OS | Ubuntu ≥ 22.04（**不支持 macOS 跑 GPU 管线**） |
| 软件 | Python 3.13、Docker + BuildKit、NVIDIA Container Toolkit、Pixi |
| 凭证 | Hugging Face token（InternVideo2）、NGC API key（CUDA 基镜像）；可选 Gemini / OpenAI 做 caption / enhance / embedding |

## 对 wiki 的映射

- 实体：[`wiki/entities/cosmos-curator.md`](../../wiki/entities/cosmos-curator.md)
- 文档摘录：[`sources/sites/cosmos-curator-docs.md`](../sites/cosmos-curator-docs.md)
- 平台：[`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)
- Cookbook Curator 配方：[`wiki/entities/cosmos-cookbook.md`](../../wiki/entities/cosmos-cookbook.md)
- 技术地图：[`wiki/overview/nvidia-physical-ai-toolchain-technology-map.md`](../../wiki/overview/nvidia-physical-ai-toolchain-technology-map.md)
