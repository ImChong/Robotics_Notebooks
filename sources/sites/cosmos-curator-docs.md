# Cosmos Curator（LHA 托管服务文档）

> 来源归档

- **标题：** Cosmos Curator — Introduction
- **类型：** site（NVIDIA 官方托管服务文档）
- **来源：** NVIDIA
- **链接：** <https://docs.nvidia.com/cosmos-curator-lha/current/introduction.html>
- **对应开源仓：** <https://github.com/NVIDIA/cosmos-curator>
- **入库日期：** 2026-09-06
- **一句话说明：** **DGX Cloud 上的 GPU 加速视频策展 SaaS**：自动切语义一致 clip、生成 embedding 与文本 prompt；输入走 S3 双桶或 ZIP，操作走 NGC WebUI 或 API。
- **沉淀到 wiki：** 是 → [`wiki/entities/cosmos-curator.md`](../../wiki/entities/cosmos-curator.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **托管服务** | 文档描述 **云端策展服务**（非纯静态文档站）；需 NVIDIA DGX Cloud / NGC 账号 |
| **自托管替代** | 同一能力的开源实现见 [NVIDIA/cosmos-curator](https://github.com/NVIDIA/cosmos-curator) → **已开源** |
| **数据入口** | AWS S3 输入/输出双桶，或 ZIP 上传（策展结果存 DGX Cloud） |

## 页面要点（introduction.html，2026-09-06）

### 两种使用方式

| 方式 | 说明 |
|------|------|
| **UI** | NGC WebUI 上传数据集并配置策展流程 |
| **API** | 程序化调用 Cosmos Curator API |

### 策展能力

- 将各种长度视频 **自动切分为语义一致 clip**
- 为每个视频生成 **embedding**
- 为 clip 生成 **文本 prompt（caption）**

### 架构（文档 Pipeline Overview）

- 管线读写 **DGX Cloud 或 S3** 上的视频与 metadata
- **Ray** 多节点多 GPU 扩展；各计算阶段 GPU 加速（NVIDIA 库）
- **阶段级 autoscaling**：例如 caption 阶段吞吐低时自动增 worker，避免瓶颈

## 与开源仓的关系

- LHA 文档面向 **托管部署**；README 与 `docs/client/end-user-guide.md` 覆盖 **本地 Docker / Slurm / NVCF** 自托管。
- 开源 split-annotate / dedup / shard-dataset 管线与托管服务能力对齐；配置模板见仓内 `examples/osmo/`。

## 对 wiki 的映射

- 实体：[`wiki/entities/cosmos-curator.md`](../../wiki/entities/cosmos-curator.md)
- 仓归档：[`sources/repos/nvidia_cosmos_curator.md`](../repos/nvidia_cosmos_curator.md)
- 平台：[`wiki/entities/nvidia-cosmos.md`](../../wiki/entities/nvidia-cosmos.md)
