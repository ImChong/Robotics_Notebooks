# Light-O1（lightorigins/Light-O1）

- **URL：** <https://github.com/lightorigins/Light-O1>
- **组织：** lightorigins
- **关联项目页：** [Light-O1 项目页](../sites/light-o1.md)
- **关联微信发布：** [wechat_lightorigins_light_o1_2026-09-21](../blogs/wechat_lightorigins_light_o1_2026-09-21.md)

## 一句话说明

Light-O1 官方推理与部署仓库：**Light-O1-Preview** 文本→全身人类动作生成（`(frames, 138)` @ 20 FPS）；含 CLI / Python API / Web console / GEAR-SONIC G1 仿真示例。

## 开源范围（入库日 2026-09-21）

| 组件 | 状态 |
|------|------|
| 推理代码 `light-deploy` | **已开源** |
| Light-O1-Preview 权重 | **HF 发布**（独立 license） |
| 完整 Light-O1 预训练权重 | **未公开** |
| loco-manipulation 真机策略 | **未公开** |

## 运行入口（README 摘要）

```bash
git clone https://github.com/lightorigins/Light-O1.git
cd Light-O1 && uv sync --extra inference
uv run --extra inference light-deploy \
  --model /path/to/Light-O1-Preview \
  --prompt "a person waves with the right hand" \
  --thinking --output human_action.npy
```

- **要求：** Linux x86-64、Python 3.11、NVIDIA CUDA 13 GPU
- **架构：** checkpoint `architectures` = `Qwen3_5ActionForConditionalGeneration`
- **后端：** vLLM（默认）或 transformers

## 交叉链接

- [Light-O1 实体页](../../wiki/entities/light-o1.md)
- [Light-O1 项目页](../sites/light-o1.md)
