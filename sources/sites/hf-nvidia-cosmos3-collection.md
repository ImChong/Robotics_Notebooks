# Hugging Face Collection — NVIDIA Cosmos3

> 来源归档

- **标题：** Cosmos3（Hugging Face Collection）
- **类型：** site / huggingface-collection
- **URL：** <https://huggingface.co/collections/nvidia/cosmos3>
- **机构：** NVIDIA
- **入库日期：** 2026-09-05
- **集合更新：** 2026-08-28（API `lastUpdated`）
- **一句话说明：** NVIDIA 官方 **Cosmos 3 全模态世界模型** 权重与交互 demo 的 Hugging Face 集合入口；描述为 *Omnimodal World Models for Physical AI*。

## 集合元数据（2026-09-05 API 核查）

| 字段 | 值 |
|------|-----|
| **slug** | `nvidia/cosmos3-69ab2f273c55ae147e43c342` |
| **描述** | Omnimodal World Models for Physical AI |
| **门控** | 集合本身 `gating: false`；各模型卡以仓上标注为准 |
| **点赞** | ~195 upvotes |

## 集合内条目（4 项）

| 条目 | 类型 | 参数量 | 门控 | 备注 |
|------|------|--------|------|------|
| [nvidia/Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super) | model | ~64.6B | 否 | 前沿全模态；数据中心 / 教师 |
| [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) | model | ~15.8B | 否 | 默认研究与部署入口 |
| [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) | model | ~3.9B | 否 | 边缘实时；Jetson / RTX Pro |
| [nvidia/Cosmos3-Action-Viewer](https://huggingface.co/spaces/nvidia/Cosmos3-Action-Viewer) | space | — | — | Viser 交互可视化；L40S 托管 |

> **注意：** README 还列出 **Cosmos3-Super-Text2Image**、**Super-Image2Video**、**Super-*-4Step** 蒸馏学生、**Cosmos3-Nano/Edge-Policy-DROID** 等 checkpoint，它们 **不在本 collection 内**，需从 [NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) 模型表或各独立 HF 仓拉取。

## 开放边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（权重 + Space demo；许可以各模型卡为准） |
| **代码** | <https://github.com/NVIDIA/cosmos> |
| **训练框架** | <https://github.com/NVIDIA/cosmos-framework> |
| **项目页** | <https://research.nvidia.com/labs/cosmos-lab/cosmos3/> |
| **许可** | 论文 / 产品 FAQ：**OpenMDW-1.1**（Cosmos 3）；具体 checkpoint 以卡上 LICENSE 行为准 |

## 典型用途

1. **快速拉权重** — `huggingface-cli download nvidia/Cosmos3-Nano` 等，接 Diffusers `Cosmos3OmniPipeline` 或 vLLM-Omni / SGLang。
2. **能力预览** — Action Viewer Space 浏览动作条件 rollout，不必本地起全栈。
3. **与 Physical AI 数据配合** — 合成数据训练见 [Physical AI HF 集合](./hf-nvidia-physical-ai-collection.md)；平台总览见 [NVIDIA Cosmos](../../wiki/entities/nvidia-cosmos.md)。

## 对 wiki 的映射

- [cosmos-3](../../wiki/entities/cosmos-3.md) — Cosmos 3 主实体页
- [nvidia-cosmos](../../wiki/entities/nvidia-cosmos.md) — 1.0 / 2.5 / 3.0 平台总览
- [nvidia_cosmos](../repos/nvidia_cosmos.md) — GitHub 仓归档
