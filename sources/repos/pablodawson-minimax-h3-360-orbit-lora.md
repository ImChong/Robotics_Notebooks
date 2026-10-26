# MiniMax-H3 360° Orbit LoRA（Pablo Dawson）

> 来源归档

- **标题：** MiniMax-H3 360° Orbit LoRA (first–last frame)
- **类型：** model / repo（Hugging Face 权重仓库）
- **发布者：** Pablo Dawson
- **模型仓库：** <https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA>
- **基座：** <https://huggingface.co/MiniMaxAI/MiniMax-H3>（FL2VA）
- **许可：** MiniMax H3 Community License Agreement；非 MIT
- **入库日期：** 2026-10-26
- **开放核查：** **部分开放** — LoRA 权重、模型卡与示例片段可获取；训练数据本身未随模型仓库发布，训练配置有记录。
- **一句话说明：** 该 LoRA 将同一张图作为首帧和尾帧条件，驱动 MiniMax-H3 FL2VA 生成静止场景的连续 360° 镜头环绕，并回到输入视角，可用于首尾相接的环绕片段。

## 训练与推理记录

- 模型卡描述的数据为 28 条人类 Gaussian Splatting 场景轨道，每条 768×768、73 帧、24 FPS；每条轨道使用同一条提示词。
- 训练使用 ostris/ai-toolkit 的 MiniMax-H3 扩展，rank/alpha 16，训练 3000 步，batch size 1；模型卡报告使用 1 张 A100 80GB，耗时约 10 小时 24 分。
- 推荐设置为 768×768、73 帧、28 steps、LoRA strength 1.0、关闭音频；首尾帧填入同一张图。模型卡注明无需 CFG 或负向提示词。
- 模型卡的对照展示：只给首帧时，基座镜头可能漂移且不回到起始视角；首尾帧相同则基座容易停成静帧；加载 LoRA 后可完成环绕并返回首帧。

## 局限与许可

- 训练集仅 28 条人类主体、正方形短片；其他主体、宽高比与长度未经验证，模型卡也提到仍可能出现眨眼或细微动作。
- 数据来自网上收集的人类 Gaussian splats；模型仓库没有提供完整数据集，训练权重不能视为可复现训练数据。
- 许可为 MiniMax H3 Community License Agreement；商用与再分发前应阅读完整上游许可。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [MiniMax H3 360° Orbit LoRA 实体](../../wiki/entities/minimax-h3-360-orbit-lora.md) | 首尾帧环绕机制与训练边界 |
| [MiniMax H3](../../wiki/entities/minimax-h3.md) | FL2VA 基座 |
| [MATLOWAI ORB360 + CardSpin](matlowai-minimax-h3-orb360-cardspin.md) | 同为环绕适配器，但基于 Ref2VA，权重与训练设计独立 |

## 一手资料

- [模型卡与模型文件](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA)
- [模型仓库提交记录](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/commits/main)
- [MiniMax H3 官方基座模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)
