# 一个大模型到底吃多少显存？如何计算？

> 来源归档（blog / 微信公众号）

- **标题：** 一个大模型到底吃多少显存？如何计算？
- **类型：** blog
- **作者：** 待核实（WebFetch 未解析公众号 nick_name；正文为配套「LLM 推理显存计算器」的公式拆解）
- **原始链接：** https://mp.weixin.qq.com/s/BzDv1uDIfGlKPw90F_e3Cw
- **入库日期：** 2026-10-01
- **抓取方式：** WebFetch 直拉 `mp.weixin.qq.com` 正文（Camoufox 快照缺失，`wechat-article-for-ai` 三次失败）
- **原始抓取落盘：** [`sources/raw/wechat_llm_inference_vram_estimation_2026-10-01.md`](../raw/wechat_llm_inference_vram_estimation_2026-10-01.md)
- **一句话说明：** 把 LLM 自回归推理 VRAM 拆成权重、KV cache、激活、CUDA 固定开销与框架因子五块；给出 GQA/MQA、MLA、MoE、SwiGLU、张量并行与 vLLM 碎片系数的逐层公式，并以 LLaMA-3 8B 与 DeepSeek-V3 671B 手算对照；声称估算误差约 ±15–20%，生产建议再留 10–15% 余量并用 `nvidia-smi` 实测。
- **步骤 2.5（开源核查）：** 科普 + 在线计算器叙述，**无**单一官方 GitHub/项目页可核；文中「计算器」为作者配套工具，入库日未在正文提取到可点击 URL。

## 核心摘录（归纳，非全文）

### 为何不能只看参数量

- 推理瓶颈常在 **显存容量** 而非 FLOPs。
- 同参数量下：量化（FP16/INT8/INT4）、**batch × 上下文** 的 KV cache、GQA/MQA/MLA、MoE **全量权重加载**、框架（vLLM vs HF）与 TP 切分都会改变每卡占用。

### 显存五块

| 块 | 要点 |
|----|------|
| 模型权重 | 逐层 Q/KV/O/FFN/Norm + embedding；MoE 须加载 **全部专家** |
| KV cache | 随 `batch × (seq_len + max_gen)` 线性增；长上下文可超过权重 |
| 激活 | 推理远小于训练；大 batch + FlashAttention 临时缓冲仍要估 |
| CUDA 固定 | 每卡 ~0.6–1.3 GB（与总参数量分档），**TP 不分摊** |
| 框架开销 | 乘性 `overheadFactor` + KV `kvFragmentFactor`（如 vLLM 1.12） |

### 关键公式（标准 decoder）

- **权重/层：** Q/O 用 `numHeads × headDim × hidden`（Gemma-2 等不可简化为 `hidden²`）；KV 标准 GQA 为 `2 × numKVHeads × headDim × hidden`；MLA 为 `hidden × kvLatentRank + 2 × kvLatentRank × numHeads × headDim`；SwiGLU FFN 为 `3 × hidden × intermediate`。
- **KV/token：** 标准 `2 × layers × numKVHeads × headDim × kv_dtype_bytes`；MLA 为 `layers × kvLatentRank × kv_dtype_bytes`（无 ×2）。
- **合计：** `per_gpu = (weight + act + kv) × overhead / tp + cuda_fixed`；`all_gpu = scalable + cuda_fixed × tp`。

### 实战读数（文中手算）

- **LLaMA-3 8B FP16 vLLM batch=1 seq=4096 gen=2048：** 约 **14.5 GB/卡** → 24 GB 消费级可跑。
- **DeepSeek-V3 671B MoE+MLA TP=8：** 权重 ~1317 GB scalable；MLA 使 KV 相对标准 GQA 约 **69×** 更小；单卡需求 ~182 GB → 需多卡 H200/H100 或 INT4。

### 局限（作者自述）

- 未建模 Pipeline/Expert Parallel、MLA 的 Q 低秩等细节；RoPE 表/tokenizer 可忽略。
- 精度 **±15–20%**；生产用 `calculated × 1.1–1.15` 并实测。

## 对 wiki 的映射

- [LLM 推理显存估算](../../wiki/concepts/llm-inference-vram-estimation.md)（新建概念页）
- [VLA 真机部署指南](../../wiki/queries/vla-deployment-guide.md) — 端侧/云端 VLA 与 LLM serving 的显存预算
- [国内 GPU 云平台选型](../../wiki/comparisons/china-gpu-cloud-platforms.md) — 租卡前估算每卡 VRAM
- [LoRA](../../wiki/concepts/lora.md) / [QLoRA 行](../../wiki/concepts/lora.md) — 训练侧 PEFT 显存（与本文推理式互补）

## 当前提炼状态

- [x] 正文抓取与 raw 归档
- [x] 概念页提炼（公式 + 流程图 + 具身/VLA 读法）
- [x] 交叉更新部署与租卡对比页
