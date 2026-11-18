# Invoker-Render 1.0 · Hugging Face 模型卡核对

> 原始模型分发资料；核查日期：2026-11-18

- **模型：** inclusionAI/invoker-render-1.0
- **Hugging Face：** https://huggingface.co/inclusionAI/invoker-render-1.0
- **对应项目：** https://www.inclusion-ai.org/CodeWorldModel/ （[归档](../sites/inclusionai-codeworldmodel.md)）
- **推理源码：** https://github.com/inclusionAI/CodeWorldModel （[归档](../repos/inclusionai-codeworldmodel.md)）
- **类型：** model（因果流式 DiT 公开权重）
- **核对范围：** 通读模型卡的特点、权重说明、运行前置、输入条件、推理入口及许可段；核对模型页显示的参数规模和张量类型。未下载 132.5 GB 权重或验证加载。
- **通读：** 已完成

## 模型文件与依赖

- 根目录提供合并的**因果流式 DiT**，约 **33.12B 参数**，FP32、29 个 safetensors 分片、638 个张量，总体约 132.5 GB。
- `bidirectional_lora/teacher-live.safetensors` 为双向 teacher 的 LoRA（约 2.3 GB），**不可再叠加至根目录已合并的因果主权重**。
- 权重仓库没有包含匹配的 VAE、Qwen、因果解码器、silence tensors 和一键 bundle。
- 文件推理首帧 1344×768、Proxy 24 FPS、`17*N+5` 帧（至少 39），与官方推理仓库的 native contract 共同使用。

## 许可与局限

HF 模型卡声明 `apache-2.0-and-cc-by-nc-4.0`；不能仅凭 GitHub 源码的 Apache-2.0 推断模型、第三方组件和最终商用许可。模型卡当日未提供托管推理服务；本归档不声称权重已经实机验证。

- **Wiki 映射：** [Code World Model 2.0 / Invoker-Render](../../wiki/entities/code-world-model-2-invoker-render.md)
