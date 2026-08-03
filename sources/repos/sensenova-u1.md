# OpenSenseNova/SenseNova-U1

> 来源归档（ingest）

- **标题：** SenseNova-U1 — 原生统一多模态模型官方开源仓（含 U1.5 Preview 发布文档）
- **类型：** repo
- **组织：** 商汤科技（SenseTime / SenseNova）
- **代码：** <https://github.com/OpenSenseNova/SenseNova-U1>
- **本次入口文档：** [`docs/u1.5_preview.md`](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/u1.5_preview.md)（`2026-07-31` 发布 U1.5-8B-MoT Preview）
- **权重：** <https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview>（归档见 [huggingface-sensenova-u1-5-8b-mot-preview.md](../sites/huggingface-sensenova-u1-5-8b-mot-preview.md)）；国内镜像见 [modelscope-sensenova-u1-5-8b-mot-preview.md](../sites/modelscope-sensenova-u1-5-8b-mot-preview.md)
- **技术报告：** [`docs/pdf/SenseNOVA_U1.pdf`](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/pdf/SenseNOVA_U1.pdf)（U1，`2026-05-10`）；arXiv [2605.12500](https://arxiv.org/abs/2605.12500)
- **架构博客：** [NEO-unify（Hugging Face Blog）](https://huggingface.co/blog/sensenova/neo-unify)
- **License：** **Apache-2.0**（仓内 `LICENSE` 全文核对）
- **在线体验：** [SenseNova-Studio](https://unify.light-ai.top/)（免费 playground；其 U1-Fast 为步数/CFG 蒸馏版，专供信息图）
- **入库日期：** 2026-08-03
- **一句话说明：** SenseNova U1 系列（**无编码器、无 VAE** 的原生统一多模态模型）官方仓：含 **推理 / 预训练 / 全参微调代码**、多个 8B-MoT 与 A3B-MoT 权重指针，以及本次 ingest 的 **U1.5 Preview** 文档——把 U1 的 patchwise MLP 像素头换成 **ConvDecoder**，主打 **原生 4K 生成** 与 **编辑时的主体/结构保持**。

## 开源核查（2026-08-03）

| 项 | 状态 |
|----|------|
| **推理代码** | **已开源** · [`examples/t2i/inference.py`](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/examples/t2i/inference.py)、[`examples/editing/inference.py`](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/examples/editing/inference.py)（U1.5 直接复用同一入口） |
| **预训练代码** | **已开源（U1.5 生成侧）** · 启动脚本 `training/shell/train_u1/U1.5_8B.sh` + 配置 `training/configs/sensenovavl_qwen3_gen/sensenovau1_5_8b_mot_pt.py`（`2026-07-31` 随 Preview 一并放出） |
| **全参微调代码** | **已开源** · [`training/README.md`](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/training/README.md)（`2026-05-21`） |
| **模型权重** | **已开源** · HF `sensenova/*`（U1.5 Preview、U1-8B-MoT、U1-A3B-MoT、Infographic V1–V3、Interleaved、8-step LoRA 等） |
| **ConvDecoder 实现** | **已开源** · [`src/sensenova_u1/models/neo_unify/modeling_fm_modules.py`](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/src/sensenova_u1/models/neo_unify/modeling_fm_modules.py) |
| **训练数据** | **未开源** — 文档仅描述「清洗 / 过滤 / 合成编辑语料、平衡中英」，无数据集发布 |
| **U1.5 技术报告** | **待发布** — 现有 PDF / arXiv 条目对应 **U1**；U1.5 仅有 `docs/u1.5_preview.md` |
| **License** | **Apache-2.0**（代码与权重仓一致，无额外营收/署名条款） |

## 入口速查（对齐 README 与 u1.5_preview.md）

| 路径 / 链接 | 作用 |
|-------------|------|
| `docs/u1.5_preview.md` | **本次 ingest 主文档**：改动动机、评测、showcase、最佳实践、Quick Start |
| `docs/parameter_breakdown.md` + `scripts/inspect_model_params.py` | 官方澄清 `8B-MoT` 命名：**≈8B 理解 + ≈8B 生成**，实测总参 **17.552B** |
| `docs/installation.md` | uv 环境（Python **3.11** / PyTorch **2.8** / CUDA **12.8**），FlashAttention 可选 |
| `docs/deployment.md`、`docs/inference_infra.md` | LightLLM + LightX2V 生产部署与性能剖析 |
| `docs/base_vs_distill.md` | 8-step 蒸馏 LoRA 与基座对比（`--cfg_scale 1.0 --num_steps 8`） |
| `docs/u1_infographic_model.md` | Infographic 专用模型系列（V1 → V3，含密集小字修正） |
| `src/sensenova_u1.5/prompt-enhancement-skill/SKILL.md` | **PE Skill**（Agent Skills 格式）：短 brief → 紧凑 Render JSON prompt |
| `src/sensenova_u1_5/caption/caption.py` | 参考图 → 结构化 JSON prompt + 自然语言 prompt（需 `OPENAI_API_KEY`） |
| `src/sensenova_u1_5/edit/edit_pe.py` | **Editing PE**：多图 + 原指令 → 改写后的模型面向指令（不改像素） |
| [SenseNova-Skills](https://github.com/OpenSenseNova/SenseNova-Skills) | 官方推荐的 agent 接入方式（见 [sensenova-skills.md](sensenova-skills.md)） |

## U1.5 相对 U1 的三处改动（README 归纳）

1. **Patchwise → Patch-Joint 重建**：U1 的 MLP 像素头逐 patch 独立回归 RGB，高分辨率下暴露 token 边界（网格纹、接缝、纹理断裂）。U1.5 改为 **ConvDecoder 渐进空间重建**——视觉 token reshape 成二维特征网格，经多级 **Pixel Shuffle** 上采样，中间插 `3×3` 卷积让相邻 patch 交互。官方称该设计在**整个预训练期**抑制网格伪影，并降低下游微调时伪影复现的概率。
2. **指令跟随 → 视觉保持**：U1 主要依赖公开编辑数据集（噪声监督、合成/修图痕迹多）。U1.5 大规模清洗与合成编辑语料，平衡中英，扩展**单图与多图参考**设定。官方结论：encoder-free 的 NEO-unify **仅用一条参考图 token 序列** 即可兼顾指令跟随与主体/结构保持，推理效率也更高。
3. **格式暴露 → 跨任务泛化**：生成/编辑语料里 JSON 格式 prompt **很少且简单**，复杂 JSON 只出现在理解侧语料中（内容与目标都不同）；但模型仍能泛化到长而结构化的生成/编辑指令——被作为「统一建模让理解侧的结构化理解与视觉规划迁移到生成侧」的证据。

## 关键评测（`docs/u1.5_preview.md`）

**Qwen-Image Bench（T2I，Overall EN / ZH）**

| 模型 | EN | ZH |
|------|----|----|
| U1 | 48.28 | 45.99 |
| **U1.5-Preview** | **49.93** | **50.25** |
| **U1.5-Preview†（+PE）** | **55.17** | **55.22** |
| Qwen Image 2\* | 55.69 | 55.63 |
| GPT Image 1 | 54.24 | 54.07 |
| GPT Image 2 | 65.23 | 64.69 |

> † = 使用 Prompt Enhance（按 Cosmos3 upsampling 策略实现）；\* = 官方自测结果。**注意 ZH 提升幅度（45.99 → 50.25）明显大于 EN**，与「加强中英文字渲染」的定位一致。

**图像编辑**

| 模型 | ImgEdit-Bench | GEdit-Bench EN G_O | GEdit-Bench CN G_O | WeEdit IA | WeEdit TC | WeEdit BP | WeEdit 均值 |
|------|---------------|--------------------|--------------------|-----------|-----------|-----------|-------------|
| U1 | 3.90 | 7.470 | 7.420 | 5.729 | 6.604 | 7.157 | 6.497 |
| **U1.5-Preview** | **4.37** | **8.172** | **8.051** | **6.532** | **7.271** | 6.752 | **6.852** |
| Qwen-Image-Edit-2511 | 4.51 | 7.877 | 7.819 | 3.180 | 3.930 | 4.630 | 3.913 |
| Nano-Banana-Pro | 4.37 | 7.738 | 7.799 | 8.580 | 9.100 | 8.850 | 8.843 |
| GPT-Image-1.5 | — | — | — | 6.520 | 7.780 | 6.150 | 6.817 |

> U1.5 在 **GEdit-Bench 中英双语均为表内最高**（8.172 / 8.051），但 **WeEdit BP 反而低于 U1**（6.752 vs 7.157）；WeEdit 综合仍显著落后 Nano-Banana-Pro。

## 官方声明的局限（Known Limitations）

- 短/欠约束 prompt 会**凭空生成文字**；
- **密集或长文本出错**，尤其小字号与中英混排；
- **复杂版式跟随不全**（精确计数、对齐、层级结构）；
- **小脸、手、肢体与细粒度物件不稳定**；
- **复杂编辑漂移**，尤其宽泛、多轮或多参考图指令。

仓库 README 另列 U1 系列整体的 Ongoing Improvements：理解侧上下文仅 **32K**；人体细节；文本渲染对 prompt 措辞敏感；交错生成仍为实验特性，RL 未针对编辑/推理/交错任务专门优化（Beta，表现与 SFT 模型相当）。

## 工程与部署要点

| 项 | 内容 |
|----|------|
| **参考推理配置** | `cfg_scale=4.0`、`timestep_shift=3.0`、`num_steps=50`（U1.5 官方值） |
| **T2I 命令** | `python examples/t2i/inference.py --model_path sensenova/SenseNova-U1.5-8B-MoT-Preview --width 2048 --height 2048 --device_map auto` |
| **编辑命令** | `python examples/editing/inference.py --image input.png --prompt ... [--use-edit-pe]`（Editing PE **默认关闭**） |
| **U1.5 预训练** | `cd training && bash shell/train_u1/U1.5_8B.sh`（先配好 model / tokenizer / data 路径） |
| **生产推理栈** | **LightLLM（理解）+ LightX2V（生成）** 解耦部署；单机 `TP2 + CFG2` 下 2048×2048 约 **0.15 s/step、端到端 ~9 s**（H100 / H200）；FA3 hybrid-mask attention 相对 Triton baseline prefill 提速 **~2.4–3.2×** |
| **一键镜像** | `docker pull lightx2v/lightllm_lightx2v:20260407` |
| **低显存** | GGUF 量化权重 + 分层加载 VRAM 模式（社区 [@smthem](https://huggingface.co/smthem/SenseNova-U1-8B-MoT-Merger-gguf) 贡献的是 **U1-8B-MoT-Merger**，非 U1.5） |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [SenseNova-U1.5](../../wiki/entities/sensenova-u1-5.md) | 本次升格的模型实体页 |
| [SenseNova-Skills](../../wiki/entities/sensenova-skills.md) | 官方推荐的 agent 接入路径；`sn-image-base` / `sn-infographic` 即以 U1 系列为后端 |
| [生成式视觉预训练](../../wiki/concepts/generative-vision-pretraining.md) | 「生成目标本身孕育理解能力」的另一条证据链（NEO-unify 双通路共演化） |
| [统一多模态 Token](../../wiki/methods/unified-multimodal-tokens.md) | VLA 侧的统一序列建模对照：U1.5 用 **连续像素 flow matching**，VLA 多用 **离散动作 token** |

## 对 wiki 的映射

- 权重页：[`sources/sites/huggingface-sensenova-u1-5-8b-mot-preview.md`](../sites/huggingface-sensenova-u1-5-8b-mot-preview.md)
- ModelScope 镜像：[`sources/sites/modelscope-sensenova-u1-5-8b-mot-preview.md`](../sites/modelscope-sensenova-u1-5-8b-mot-preview.md)
- 技能库归档（既有）：[`sources/repos/sensenova-skills.md`](sensenova-skills.md)
- 沉淀 **[`wiki/entities/sensenova-u1-5.md`](../../wiki/entities/sensenova-u1-5.md)**
