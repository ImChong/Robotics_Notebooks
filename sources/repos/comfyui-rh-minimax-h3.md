# ComfyUI-RH-MiniMax-H3（RunningHub 官方插件）

> 来源归档

- **标题：** ComfyUI-RH-MiniMax-H3
- **类型：** repo
- **组织：** RunningHub（MiniMax **合作伙伴**；README 自述开发维护）
- **代码：** <https://github.com/RH-RunningHub/ComfyUI-RH-MiniMax-H3>
- **许可：** Apache-2.0
- **入库日期：** 2026-09-29
- **一句话说明：** 在 **ComfyUI 进程内原生运行 MiniMax-H3**（无 SGLang / 非 Diffusers 管线）：**T2VA、FL2VA、Ref2VA、V2A**；**INT8 ConvRot** 权重 + 可选 turbo LoRA + offload；主节点 `RHMiniMaxH3VideoGen` / `RHMiniMaxH3RefGen`；**单卡 24GB** 可跑（README 与社区文档口径）。
- **步骤 2.5：** **已开源** — 插件 Apache-2.0；权重来自 [Gluttony10/MiniMax-H3-INT8-CONVROT](https://huggingface.co/Gluttony10/MiniMax-H3-INT8-CONVROT)（约 **95 GiB** 包，非 MiniMax 官方 HF 全精度仓）。

## 模型目录（默认）

`ComfyUI/models/MiniMax-H3-INT8-CONVROT/` — FL2VA/Ref2VA DiT、Qwen3-VL-32B INT8、video/audio VAE、turbo LoRA 等（见 README 树状结构）。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [MiniMax H3 实体](../../wiki/entities/minimax-h3.md) | 模型与任务说明 |
| [ComfyUI 实体](../../wiki/entities/comfyui.md) | 宿主引擎 |
| [RunningHub RH Enhanced Camp](../sites/runninghub-rh-enhanced-h3-camp.md) | 云端封装工作流（workflowId 见该页） |
