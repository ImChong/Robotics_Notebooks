# MoonshotAI/Kimi-K3

> 来源归档（ingest）

- **标题：** Kimi K3 — Open Frontier Intelligence（官方 GitHub）
- **类型：** repo
- **组织：** 月之暗面（Moonshot AI）
- **代码：** <https://github.com/MoonshotAI/Kimi-K3>
- **权重：** <https://huggingface.co/moonshotai/Kimi-K3>（主发布）；ModelScope 镜像见 [modelscope-moonshotai-kimi-k3.md](../sites/modelscope-moonshotai-kimi-k3.md)
- **技术报告：** [`k3_tech_report.pdf`](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)（约 2.5MB / 47 页）
- **License：** **Kimi K3 License**（Modified MIT 风格；含 MaaS 营收门槛与大产品署名条款）— <https://github.com/MoonshotAI/Kimi-K3/blob/main/LICENSE>
- **技术博客：** <https://www.kimi.com/blog/kimi-k3>
- **入库日期：** 2026-07-27
- **一句话说明：** Kimi K3 官方开源入口仓：README（模型卡摘要 / 评测 / 部署指针）、**技术报告 PDF**、**Kimi K3 License**；完整 **MXFP4** 权重在 Hugging Face / ModelScope，本仓**不含**训练代码与权重分片。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| **模型权重** | **已开源（开放权重）** · HF `moonshotai/Kimi-K3`（96× safetensors，合计约 **1.56 TB**）+ ModelScope 镜像 |
| **技术报告** | **已发布** · 仓内 `k3_tech_report.pdf`（无 arXiv 条目，截至入库日） |
| **推理生态** | README 推荐 **vLLM** / **SGLang** / **TokenSpeed** 官方 recipe |
| **本仓代码** | 仅 `LICENSE` + `README.md` + `assets/` + `k3_tech_report.pdf` — **无可运行训练 / 推理脚本** |
| **训练数据 / 训练栈** | **未在本仓发布**（License 文案涵盖“training code”，但公开树中无实现） |
| **许可证** | **Kimi K3 License**：研究 / 自托管 / 微调可用；**MaaS** 且连续 12 月营收 **>\$20M** 须另签协议；月活 **>1 亿** 或月营收 **>\$20M** 的商业产品须显著展示 “Kimi K3” |

## 入口速查（对齐 README）

| 路径 / 链接 | 作用 |
|-------------|------|
| `README.md` | 模型介绍、规格表、评测表、MXFP4、部署与 Usage |
| `k3_tech_report.pdf` | 完整技术报告（架构 / 预训练 / 后训练 / 基础设施 / 评测 / 案例） |
| `LICENSE` | Kimi K3 License 全文 |
| [HF `moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) | 权重 + `configuration_kimi_k3.py` / `modeling_kimi_k3` 等 custom code |
| [vLLM recipes](https://recipes.vllm.ai/moonshotai/Kimi-K3) | 推荐推理引擎之一 |
| [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) | 推荐推理引擎之一 |
| [Kimi Code CLI](https://www.kimi.com/code) | 官方推荐 coding agent harness（`/model` 选 K3） |

## Model Summary（README 规格表摘录）

| 项 | 值 |
|----|-----|
| Architecture | MoE |
| Total / Activated params | **2.8T** / **104B** |
| Layers | 93（含 1 dense） |
| Attention | **69 KDA + 24 Gated MLA** |
| Experts | 896 routed，每 token 选 **16**；shared experts **2** |
| Context | **1,048,576** tokens |
| Vision | **MoonViT-V2**（401M） |
| Quantization | **MXFP4** weights / **MXFP8** activations（QAT） |
| Vocab | 160K |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Kimi K3](../../wiki/entities/kimi-k3.md) | 旗舰模型实体：开源状态、部署与 harness 实践 |
| [Muon](../../wiki/methods/muon.md) | 技术报告含 **Per-Head Muon** |
| [真机策略 autoresearch 闭环](../../wiki/queries/real-robot-policy-autoresearch-harness.md) | Kimi Code + K3 作为长程 coding agent 后端选项 |
| [ENPIRE](../../wiki/methods/enpire.md) | AutoEnvBench 已覆盖 Kimi Code 产品线 |

## 对 wiki 的映射

- 权重页：[`sources/sites/huggingface-moonshotai-kimi-k3.md`](../sites/huggingface-moonshotai-kimi-k3.md)
- ModelScope：[`sources/sites/modelscope-moonshotai-kimi-k3.md`](../sites/modelscope-moonshotai-kimi-k3.md)
- 技术报告：[`sources/papers/kimi_k3_tech_report.md`](../papers/kimi_k3_tech_report.md)
- 博客 / API（既有）：[`../blogs/kimi_k3_tech_blog.md`](../blogs/kimi_k3_tech_blog.md)、[`../courses/kimi_k3_api_quickstart.md`](../courses/kimi_k3_api_quickstart.md)
- 沉淀 **[`wiki/entities/kimi-k3.md`](../../wiki/entities/kimi-k3.md)**
