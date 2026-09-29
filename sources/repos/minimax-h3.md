# MiniMax-H3（官方开源仓库）

> 来源归档

- **标题：** MiniMax-H3
- **类型：** repo
- **组织：** MiniMax AI
- **代码：** <https://github.com/MiniMax-AI/MiniMax-H3>
- **权重：** <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- **许可：** MiniMax H3 Community License（见仓内 `LICENSE`）
- **入库日期：** 2026-09-29
- **一句话说明：** **全模态音视频生成系统** 开源实现：**H3-Base**（768p 联合音视频 DiT + Qwen3-VL-32B 编码 + 双 VAE）与 **H3-Regenerate-2K**；支持 T2VA、FL2VA（首尾帧/I2V）、Ref2VA（≤9 图 / ≤3 视音频参考）；**H3-Context-IR 预处理未随仓开源**（提供官方 API 与 Prompting Guidance）。
- **步骤 2.5（开源核查）：** **已开源**（2026-09-29 GitHub/HF 复核）— 推理代码、权重、Diffusers 集成、示例脚本与 `skills/h3-prompt-writing`；**部分开源** — Context-IR、稀疏注意力实现、Regenerate-2K 模块文档写明 **尚未开源 / 后续发布**，2K 完整链路需 **开放平台 API** 或自研预处理。

## 系统模块（README）

| 模块 | 开放程度 |
|------|----------|
| **H3-Context-IR** | **未开源**；官方 [Context-IR API](https://platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir) |
| **H3-Base** | 开源；768p 联合生成 |
| **H3-Regenerate-2K** | 768p 结果 + 原上下文再生成 2K；**Regenerate 模块仓内标注待发布** |
| **稀疏注意力** | 初始发行 **仅 full attention 推理** |

## 规格摘要

- 输出：**4–15 s**、**24 FPS**、多宽高比；短边默认 **768**；**32 kHz 立体声**
- 变体：**H3-Base-FL2VA**、**H3-Base-Ref2VA**
- 在线：**hailuoai.video** / **platform.minimax.io** API

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [MiniMax H3 实体](../../wiki/entities/minimax-h3.md) | wiki 主节点 |
| [ComfyUI-RH-MiniMax-H3](comfyui-rh-minimax-h3.md) | RunningHub 本地 Comfy 插件（INT8 ConvRot） |
| [RunningHub RH Enhanced 项目页](../sites/runninghub-rh-enhanced-h3-camp.md) | 云端 Camp 工作流 |
