# Qwen-Image-2.1 官方博客（Qwen.ai）

> 来源归档（blog）

- **标题：** Qwen-Image-2.1 — 通义图像生成与编辑 2.1
- **类型：** blog / announcement
- **URL：** <https://qwen.ai/blog?id=qwen-image-2.1>
- **机构：** Qwen Team（阿里巴巴通义）
- **入库日期：** 2026-09-20
- **抓取说明：** 博客页为 CSR 渲染，正文以 [GitHub README](https://github.com/QwenLM/Qwen-Image-2.1) 与 [Hugging Face 模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1) 交叉核对后归档。
- **一句话说明：** 2026-09-20 发布 **Qwen-Image-2.1**：7B Single-Stream DiT 统一 **文生图 + 图像编辑 + 原生 RGBA 透明**；支持最多 10 张参考图、局部标注编辑；权重与代码同步开源。

## 发布要点（官方 README / 模型卡归纳）

1. **Compact and Efficient：** mixed-granularity attention + prefix KV cache reuse，在较低算力下保持生成质量。
2. **Native Transparency：** 同一模型可生成/编辑 **RGBA** 透明图，或从照片提取主体。
3. **Versatile Editing：** 最多 **10** 张参考图；圆选/涂鸦/独立 mask 局部编辑；人物与商品 **identity preservation**。
4. **Realistic Textures：** 改进排版、人像光效与细节质感。
5. **Day-0 生态：** Diffusers `QwenImage21Pipeline`、ComfyUI 原生工作流、vLLM-Omni / SGLang / LightX2V 推理加速。

## 对 wiki 的映射

- [qwen-image-2-1.md](../../wiki/entities/qwen-image-2-1.md)
- [qwen-image-2-1.md](../repos/qwen-image-2-1.md) — 仓库与权重归档
