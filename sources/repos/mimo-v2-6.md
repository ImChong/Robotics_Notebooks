# MiMo-V2.6 开源资源索引

> 来源归档

- **标题：** MiMo-V2.6 Open Source Collection
- **类型：** repo（HF 模型集合 + GitHub 工程栈）
- **HF 集合：** <https://huggingface.co/collections/XiaomiMiMo/mimo-v26>
- **组织：** [XiaomiMiMo](https://huggingface.co/XiaomiMiMo) · [GitHub XiaomiMiMo](https://github.com/XiaomiMiMo)
- **入库日期：** 2026-09-23
- **一句话说明：** MiMo-V2.6 **Pro / Flash RL 权重**、**Distill-Qwen-9B**、技术报告 PDF，以及 **verl / uni-agent / ~7k RL 环境 / mini-harnesses** 的集中入口。

## 开源边界（步骤 2.5，2026-09-23）

| 资源 | 状态 | URL |
|------|------|-----|
| **MiMo-V2.6-Pro-RL** | **已开源**（MIT） | <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL> |
| **MiMo-V2.6-Flash-RL** | **已开源**（MIT） | <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL> |
| **MiMo-V2.6-Distill-Qwen-9B** | **已开源** | <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B> |
| **技术报告 PDF** | **已公开** | Pro-RL 仓 `MiMo_V2_6_technical_report.pdf` |
| **RL 训练（verl）** | **已开源** | <https://github.com/XiaomiMiMo/verl> |
| **Agent RL（uni-agent）** | **已开源**（Apache-2.0） | <https://github.com/XiaomiMiMo/uni-agent> |
| **RL 任务环境 ~7k** | **已开源** | 随报告 §7 / 集合说明发布 |
| **mini-harnesses** | **已开源** | 发布说明「轻量可组合 Harness」 |
| **MiMo-Embodied 评测** | **部分**（eval-only） | <https://github.com/XiaomiMiMo/MiMo-Embodied> — **独立**具身 VLM 评测套件，非 V2.6 训推本体 |

## HF 模型（集合内，2026-09-23）

| 模型 | 参数量（HF 元数据） | 用途 |
|------|---------------------|------|
| MiMo-V2.6-Pro-RL | ~1.02T total | 旗舰 RL checkpoint |
| MiMo-V2.6-Flash-RL | ~311B total | 高效 RL checkpoint |
| MiMo-V2.6-Distill-Qwen-9B | ~9.4B | 社区 RL 复现起点 |

## 部署入口（Pro-RL README）

- **SGLang：** [MiMo cookbook](https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.5)（V2.6 沿用）；需多节点 TP/EP（示例 `--tp 16 --ep 16`）。
- **vLLM：** [MiMo-V2.5 recipe](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5)；镜像 `vllm/vllm-openai:mimov25-cu129`。
- **API：** `platform.xiaomimimo.com` · **Desktop：** `mimo.xiaomimimo.com/desktop/`

## 相关 GitHub（维护索引）

| 仓库 | 角色 |
|------|------|
| [MiMo](https://github.com/XiaomiMiMo/MiMo) | 组织主仓 / 品牌资源 |
| [verl](https://github.com/XiaomiMiMo/verl) | RL 训练库（MiMo 定制 fork） |
| [uni-agent](https://github.com/XiaomiMiMo/uni-agent) | 长程 Agent RL：harness 接入、并发 rollout、轨迹采集 |
| [MiMo-Code](https://github.com/XiaomiMiMo/MiMo-Code) | 代码 Agent 产品栈 |
| [MiMo-Embodied](https://github.com/XiaomiMiMo/MiMo-Embodied) | 具身 / 自动驾驶 VLM **评测**（arXiv:2511.16518） |
| [awesome-mimo-agent](https://github.com/XiaomiMiMo/awesome-mimo-agent) | Agent 生态索引 |

## 对 wiki 的映射

- 主实体：[MiMo-V2.6](../../wiki/entities/mimo-v2-6.md)
- 技术报告：[mimo_v2_6_technical_report_2026.md](../papers/mimo_v2_6_technical_report_2026.md)
- 发布说明：[mimo_v2_6_release_2026-09-22.md](../blogs/mimo_v2_6_release_2026-09-22.md)
