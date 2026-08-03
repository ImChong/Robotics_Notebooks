# SenseNova/SenseNova-U1.5-8B-MoT-Preview（ModelScope）

> 来源归档（ingest）

- **标题：** SenseNova-U1.5-8B-MoT (Preview) — 魔搭社区模型镜像
- **类型：** site / model hub（ModelScope）
- **组织：** 商汤科技（SenseTime / SenseNova）
- **官方入口：** <https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-Preview>
- **主权重面（HF）：** <https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview>（归档见 [huggingface-sensenova-u1-5-8b-mot-preview.md](./huggingface-sensenova-u1-5-8b-mot-preview.md)）
- **代码仓：** <https://github.com/OpenSenseNova/SenseNova-U1>（归档见 [sensenova-u1.md](../repos/sensenova-u1.md)）
- **入库日期：** 2026-08-03
- **一句话说明：** U1.5 Preview 的**国内下载镜像**：文件树与 Hugging Face 一致（13 个分片 ≈ **50.2 GB** + `config.json` / tokenizer），任务归类为**统一多模态**（`multi-modal`）；页面为前端渲染，元数据以 ModelScope Open API 核对。

## 开源核查（2026-08-03）

| 项 | 状态 |
|----|------|
| **模型页** | **已发布** · API `Name=SenseNova-U1.5-8B-MoT-Preview` |
| **任务标签** | `multi-modal` / **统一多模态** |
| **权重** | **已镜像** · `model-0000{1,5..16}-of-00016.safetensors`（13 个文件，合计 **50,192,454,398 B ≈ 50.2 GB**），与 HF 分片编号缺号情况一致（权重完整，见 HF 归档说明） |
| **License** | **API 元数据 `License` 为空**；实际条款以 GitHub 仓 **Apache-2.0** 为准，不要因镜像页未标注就当作未授权 |
| **额外文件** | 比 HF 多一个 `configuration.json`（ModelScope 侧的框架/任务声明），其余文件同名同构 |
| **热度** | 下载 **54**、Stars **6**（入库日快照；同期 HF 为 121 / 51） |
| **用途** | 国内网络的下载通道；**选型与复现仍以 HF 模型卡 + GitHub `docs/u1.5_preview.md` 为准** |

## 使用提示

- 下载走 `modelscope` SDK 或 Git LFS 均可；拉全量需预留 **≥55 GB** 磁盘，另加解包/载入余量。
- 载入仍需 GitHub 仓的 custom code（`trust_remote_code`）与 `examples/` 推理入口——**镜像只解决权重可达性，不含代码**。
- 显存按 **bf16 约 35 GB** 估算（官方 `inspect_model_params.py` 口径），不要用 50 GB 磁盘体积直接换算。

## 对 wiki 的映射

- [SenseNova-U1.5](../../wiki/entities/sensenova-u1-5.md)
- [huggingface-sensenova-u1-5-8b-mot-preview.md](./huggingface-sensenova-u1-5-8b-mot-preview.md)
- [sensenova-u1.md（GitHub）](../repos/sensenova-u1.md)
