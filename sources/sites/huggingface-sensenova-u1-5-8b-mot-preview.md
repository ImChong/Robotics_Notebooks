# sensenova/SenseNova-U1.5-8B-MoT-Preview（Hugging Face）

> 来源归档（ingest）

- **标题：** SenseNova-U1.5-8B-MoT (Preview) — Hugging Face 模型卡与开放权重
- **类型：** site / model card（Hugging Face Hub）
- **组织：** 商汤科技（SenseTime / SenseNova）
- **官方入口：** <https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-Preview>
- **合集：** <https://huggingface.co/collections/sensenova/sensenova-u15>
- **架构博客：** <https://huggingface.co/blog/sensenova/neo-unify>
- **关联仓 / 报告 / 镜像：**
  - <https://github.com/OpenSenseNova/SenseNova-U1>（归档见 [sensenova-u1.md](../repos/sensenova-u1.md)）
  - <https://arxiv.org/abs/2605.12500>（U1 技术报告）
  - <https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-Preview>（归档见 [modelscope-sensenova-u1-5-8b-mot-preview.md](./modelscope-sensenova-u1-5-8b-mot-preview.md)）
- **入库日期：** 2026-08-03
- **一句话说明：** U1.5 Preview 的**主权重发布面**：`any-to-any` 原生统一多模态（`NEOChatModel` / `model_type: neo_chat`），BF16 + F32 混合存储、总参 **17.53B**、盘上约 **50.2 GB**，`trust_remote_code` 加载；模型卡正文与仓内 `docs/u1.5_preview.md` 基本同源，另附中文版 `README_CN.md`。

## 开源核查（2026-08-03）

| 项 | 状态 |
|----|------|
| **gated** | **否** — 无需申请，直接下载 |
| **权重** | **已开源** · 13 个 `safetensors` 分片，合计约 **50.2 GB** |
| **License 字段** | **模型卡 frontmatter 未声明 license**（HF API `license: null`）；代码与权重的实际条款以仓内 **Apache-2.0** `LICENSE` 为准 |
| **推理代码** | 需配合 GitHub 仓 `examples/`；模型卡给出 `AutoModel.from_pretrained(..., trust_remote_code=True)` 与 CLI 两条路径 |
| **训练配置** | 预训练启动脚本与 config 在 GitHub 仓（`training/`），**不在权重仓内** |
| **Hub 计数** | 创建 `2026-07-28`，最后更新 `2026-08-03`；下载 **121**、点赞 **51**（入库日快照，仍在爬坡） |

### 分片编号提示（避免误判下载失败）

文件名后缀为 `-of-00016`，但仓内**实际只有 13 个分片**（`00001`、`00005`–`00016`；`00002`–`00004` 缺号）。核对 `model.safetensors.index.json` 的 `weight_map`：**所有权重都映射到这 13 个存在的文件**，缺号分片未被引用——即 **权重完整，编号是遗留的分片规划**，不必重试下载或怀疑仓库损坏。

## 规格核对（`config.json` + HF safetensors 元数据）

| 项 | 值 | 说明 |
|----|-----|------|
| **架构类名** | `NEOChatModel`（`model_type: neo_chat`） | `AutoConfig` / `AutoModel` 走 `configuration_neo_chat.py` / `modeling_neo_chat.py`，custom code |
| **总参** | **17.53B**（BF16 **9.97B** + F32 **7.56B**） | 与官方 `docs/parameter_breakdown.md` 的 **17.552B** 一致 |
| **参数分组（官方脚本）** | 理解侧 **8.12B** / 生成侧 **8.19B** / 共享 **1.245B** | 共享部分为 `embed_tokens` + `lm_head`（文本 token I/O，两条通路复用） |
| **通路覆盖** | 理解 **9.37B**、生成 **9.43B** | 共享参数在两条通路各计一次，故比例之和 >100% |
| **语言主干** | Qwen3：**42** 层、`hidden 4096`、`intermediate 12288`、**32** 头 / **8** KV 头、`vocab 151936` | 配置目录名亦为 `sensenovavl_qwen3_gen` |
| **位置编码** | `max_position_embeddings 262144`、`rope_theta 5e6`；另有 `rope_theta_hw / max_position_embeddings_hw` | 文本与二维视觉分别走不同 RoPE 参数 |
| **视觉侧** | `NEOVisionModel`，`hidden 1024`、`patch_size 16`、`downsample_ratio 0.5` | **无独立预训练视觉编码器 / VAE**（NEO-unify 的 encoder-free 主张） |
| **像素预算** | `min_pixels 65536`、`max_pixels 16777216` | 上限恰为 **4096×4096**，对应「原生 4K 生成」的口径 |
| **生成头** | `use_pixel_head: true`、`fm_head_dim 1536`、`fm_head_layers 2`、`fm_head_mlp_ratio 1` | flow-matching 像素头；U1.5 的 ConvDecoder 在 `modeling_fm_modules.py` |
| **流匹配调度** | `time_schedule standard`、`time_shift_type exponential`、`base_shift 0.5`、`max_shift 1.15`、`P_mean -0.8`、`P_std 0.8` | 训练侧时间步/噪声调度；推理默认 `timestep_shift` 由 CLI 覆盖为 **3.0** |
| **序列长度（图像）** | `base_image_seq_len 64`、`max_image_seq_len 4096` | 配合 `noise_scale_mode: resolution` 做分辨率相关噪声缩放 |

> **BF16 + F32 混存的实际含义：** 盘上 ~50.2 GB 主要来自 F32 存储的那 7.56B 参数；官方 `inspect_model_params.py` 报告 **bfloat16 载入约 35.1 GB**。做显存预算时按 bf16 口径估算，不要直接拿磁盘体积换算。

## 模型卡要点（README 归纳）

- **定位：** 基于 [NEO-unify](https://huggingface.co/blog/sensenova/neo-unify) 的原生统一多模态模型，新增 patch 编解码层；**理解与生成共用一条序列表示**，文本走自回归交叉熵、视觉走**像素 flow matching**。
- **五条卖点：** 原生 4K 高效生成；纹理/材质/光照/真实感提升；中英文字渲染与密集复杂版式；编辑侧指令跟随 + 主体身份 + 结构一致性；**掩码 / 边界框 / 视觉标记的区域可控编辑**。
- **用法：** `cfg_scale=4.0`、`timestep_shift=3.0`、`num_steps=50`；编辑与多图参考按顺序传图并在指令中说明各图角色。
- **最佳实践（模型卡明确标注为「模型外部的可选工作流」，不应与模型裸能力混谈）：**
  - **Path A — PE Skill：** 短 brief → 紧凑 Render JSON prompt（保留主体、可见文案、数量、版式约束与排除项）；
  - **Path B — 参考图检索 + Caption-to-Prompt：** 从参考图反解版式网格、视觉焦点、字体层级、配色、材质、镜头、光照与负向约束，再换题换文案；
  - **Editing PE：** 多模态改写器把用户指令显式化（明确编辑目标与位置、给多参考图分派角色、指明哪些区域必须不变），`--use-edit-pe` 开启，**默认关闭**。
- **已知局限：** 见 [仓库归档](../repos/sensenova-u1.md#官方声明的局限known-limitations)（凭空生成文字、密集文本出错、复杂版式不全、小脸/手不稳、复杂多轮编辑漂移）。
- **口径提醒：** 模型卡自称 Preview，「更强的正式版即将发布」——选型时应把当前数字当**下界**，并留意 U1.5 技术报告尚未发布。

## 对 wiki 的映射

- [SenseNova-U1.5](../../wiki/entities/sensenova-u1-5.md) — 「核心结构 / 开源状态 / 工程实践」
- [sensenova-u1.md（GitHub）](../repos/sensenova-u1.md)
- [modelscope-sensenova-u1-5-8b-mot-preview.md](./modelscope-sensenova-u1-5-8b-mot-preview.md)
