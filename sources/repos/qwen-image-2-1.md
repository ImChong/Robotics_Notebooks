# Qwen-Image-2.1

> 来源归档（repo + weights）

- **标题：** Qwen-Image-2.1 — Qwen's most powerful open-source image generation model
- **类型：** repo + model weights
- **组织：** Qwen Team（阿里巴巴通义）
- **代码：** <https://github.com/QwenLM/Qwen-Image-2.1>
- **Hugging Face：** <https://huggingface.co/Qwen/Qwen-Image-2.1>
- **ModelScope：** <https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1>
- **Demo：** <https://huggingface.co/spaces/Qwen/Qwen-Image-2.1>
- **博客：** <https://qwen.ai/blog?id=qwen-image-2.1>
- **Prompt 重写（T2I）：** <https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I>
- **Prompt 重写（Edit）：** <https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I>
- **许可证：** Qwen Research License Agreement
- **入库日期：** 2026-09-20
- **一句话说明：** 7B Single-Stream DiT（32 层）统一文生图与图像编辑；Qwen3-VL 8B 文本/条件编码 + 64 通道 RGBA VAE（16× 压缩）；原生 2K 多宽高比；Diffusers / ComfyUI / vLLM-Omni / SGLang / LightX2V Day-0 支持。

## 步骤 2.5 开源核查（2026-09-20）

| 项 | 结论 |
|----|------|
| GitHub | [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) — README、Quick Start、`prompt_rewrite/` |
| 权重 | HF [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)；ModelScope 同名 |
| 推理入口 | `diffusers.QwenImage21Pipeline`；vLLM-Omni `examples/offline_inference/`；ComfyUI 模板工作流 |
| **判定** | **已开源** — 代码 + 权重 + 官方 prompt 重写子模型均可公开获取 |

## 架构摘要（README § Architecture）

| 组件 | 规格 |
|------|------|
| **DiT** | 32 层、**7B** 参数；single-stream；block-causal attention（文本 token 级 causal，图像 chunk 级双向） |
| **Text Encoder** | **Qwen3-VL 8B** — 统一编码文本指令与条件图像 |
| **VAE** | **64 通道 RGBA** autoencoder，**16×** 空间压缩，原生透明 |
| **Scheduler** | Flow Matching + Euler discrete + dynamic shifting |
| **加速** | mixed-granularity attention → **prefix KV cache reuse**（条件图像+文本首步编码，去噪步复用） |

## 默认推理参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `num_inference_steps` | 40 | 去噪步数 |
| `width` × `height` | 2048 × 2048 | 原生 2K；支持 1:1 / 4:3 / 3:4 / 3:2 / 2:3 / 16:9 / 9:16 |

## 核心能力摘录

1. **T2I / Edit 统一 pipeline：** 同一 `QwenImage21Pipeline`；编辑时传入 `image=`（单张或多张，最多 **10** 参考图）。
2. **RGBA 透明生成：** 推荐 prompt 前缀 `This is an RGBA image with transparency. ... The image has alpha channel and the background is transparent.`
3. **Prompt 重写：** 基于 Qwen3.5-VL 9B 微调 PE 模型（T2I / Edit 各一）；`prompt_rewrite/run_vllm.py` 批量扩展短 prompt 并输出 `wh_ratio`。
4. **内存优化：** `pipe.enable_model_cpu_offload()`；checkpoint 默认 `causal_condition: true` 自动 prefix KV cache。
5. **Day-0 生态（2026-09-20）：** Diffusers [#14804](https://github.com/huggingface/diffusers/pull/14804)；ComfyUI [workflow templates](https://github.com/Comfy-Org/workflow_templates)；vLLM-Omni / SGLang / LightX2V 加速指南。

## 对 wiki 的映射

- [qwen-image-2-1.md](../../wiki/entities/qwen-image-2-1.md) — 实体页（架构、具身相关用法、工程入口）
- 交叉：[WH0](../papers/wh0_arxiv_2606_22136.md)、[RoboEdit](../papers/roboedit_arxiv_2608_18948.md) — 机器人管线中 Qwen-Image-Edit 系前代/同类用法
- [generative-data-augmentation.md](../../wiki/methods/generative-data-augmentation.md) — 场景编辑 / 物体合成数据增强
