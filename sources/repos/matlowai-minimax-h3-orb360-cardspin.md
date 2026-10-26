# MiniMax-H3 ORB360 + CardSpin LoRA（MATLOWAI）

> 来源归档

- **标题：** MiniMax-H3 ORB360: 360 orbit + CardSpin
- **类型：** model / repo（Hugging Face 权重仓库）
- **发布者：** MATLOWAI
- **模型仓库：** <https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin>
- **基座：** <https://huggingface.co/MiniMaxAI/MiniMax-H3>（Ref2VA）
- **许可：** MiniMax H3 Community License Agreement；非 MIT
- **入库日期：** 2026-10-26
- **开放核查：** **部分开放** — LoRA 权重、模型卡、提示词与生成示例可获取；模型卡公开训练配置，但没有提供其训练数据集或完整训练代码仓库。
- **一句话说明：** 发布者提供 rank-32 的 MiniMax-H3 Ref2VA 适配器：纯 ORB360 step 1500 面向清晰顺时针环绕；CardSpin v1/v2 额外训练卡片式旋转效果，模型卡的对比显示 CardSpin 会损失细节。

## 版本与用途

| 文件 | 用途 | 模型卡说明 |
|------|------|------|
| minimax_h3_orb360_step1500.safetensors | 连续 360° 镜头环绕 | 1500 步高分辨率 Blender 轨道训练，支持标注 1–4 张参考图；作者推荐用于清晰环绕与多视角 |
| minimax_h3_orb360_cardspin_step50.safetensors | CardSpin v1 | 在较早 750 步环绕权重上追加 50 步卡片式旋转训练 |
| minimax_h3_orb360_cardspin_v2_step50.safetensors | CardSpin v2 | 在 step 1500 权重上追加 50 步卡片式旋转训练 |

## 训练与已知边界

- ORB360 step 1500：模型卡称使用 23 个物体在 Blender Cycles 中渲染 345 条顺时针轨道；按不同参考图组合形成 960 个训练条目，共 1500 步。训练使用 musubi-tuner v0.3.5、MiniMax-H3 Ref2VA BF16 基座、rank/alpha 32、单张 RTX PRO 6000 96GB。
- CardSpin v2 仅追加一条生成视频训练样例 50 步；模型卡公布的对比指标显示 CardSpin v1/v2 的细节约为 step 1500 的三分之一。
- step 1500 在灰背景 Blender 渲染和少量留出对象上测试；作者称实拍图的镜头高度控制尚未稳定。只输入一张图时，未见过的物体侧面由模型补全。
- 许可继承 MiniMax H3 Community License Agreement；商用、再分发及地域限制应按完整许可核对。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [MiniMax H3 ORB360 实体](../../wiki/entities/minimax-h3-orb360-cardspin.md) | 版本、训练配置与边界归纳 |
| [MiniMax H3](../../wiki/entities/minimax-h3.md) | Ref2VA 基座 |
| [Pablo Dawson 360 Orbit LoRA](pablodawson-minimax-h3-360-orbit-lora.md) | 同基座下采用 FL2VA 的独立适配器 |

## 一手资料

- [模型卡与文件](https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin)
- [模型文件列表](https://huggingface.co/MATLOWAI/MiniMax-H3-ORB360-CardSpin/tree/main)
- [MiniMax H3 官方基座模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)
