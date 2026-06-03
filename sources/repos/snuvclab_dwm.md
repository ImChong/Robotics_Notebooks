# snuvclab/dwm

> 来源归档

- **标题：** Dexterous World Models（官方实现）
- **类型：** repo
- **组织：** SNU VC Lab（Seoul National University）
- **代码：** <https://github.com/snuvclab/dwm>
- **论文：** <https://arxiv.org/abs/2512.17907>
- **项目页：** <https://snuvclab.github.io/dwm/>
- **入库日期：** 2026-05-17（工程栈补充：2026-06-03）
- **一句话说明：** CVPR 2026 **Dexterous World Models** 官方实现：**CogVideoX-5B LoRA** 上 static+hand 通道拼接条件训练/推理，初始化来自 **VideoX-Fun** 全掩码修复权重；2026-04-03 开源代码并同步发布 **WAN 版** 变体；数据侧期望 **交互 / 静态场景 / 手网格** 三元组 + 文本 prompt 与预编码潜变量目录结构。

## 里程碑（upstream README，2026-06-03 检索）

| 日期 | 事件 |
|------|------|
| 2026-02-21 | CVPR 2026 接收 |
| 2026-04-03 | **代码发布**（含 **DWM WAN** 版本） |

## 工程栈摘要

| 组件 | 说明 |
|------|------|
| **骨干** | CogVideoX-5B + LoRA（示例配置 `training/cogvideox/configs/examples/dwm_cogvideox_5b_lora.yaml`） |
| **条件拼接** | static scene latents + hand mesh latents 在通道维与噪声潜变量拼接（`train_static_hand_concat` / `infer_static_hand_concat`） |
| **初始化** | 预训练 **VideoX-Fun** 视频修复扩散（全 mask ≈ 恒等映射，对应论文残差动力学动机） |
| **变体** | 除 CogVideoX 路径外，仓库另含 **WAN** 版 DWM 实现（2026-04 与主代码一并发布） |
| **算力** | 默认 5B 训练路径通常需 **80 GB 级** GPU（README Notes） |
| **依赖生态** | VideoX-Fun、finetrainers、CogVideo、Wan2.1（Acknowledgements） |

## 数据与训练入口（结构级，非命令缓存）

预处理与 caption 见 `data_processing/README.md`；训练主文档 `training/cogvideox/README.md`。每个样本目录期望包含：

- `videos/` — 交互视频（ground-truth rollout）
- `videos_static/` — 同轨迹静态场景视频
- `videos_hands/` — 对齐手网格渲染视频
- `prompts/`、`prompts_rewrite/` — 文本语义条件
- `video_latents/`、`static_video_latents/`、`hand_video_latents/`、`prompt_embeds_rewrite/` — 预编码潜变量（可选加速）

论文对齐的数据划分清单：`dataset_files/trumans_train.txt`、`taste_rob_train.txt` 及对应 test split。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [DWM（Dexterous World Models）](../../wiki/methods/dwm.md) | 方法级归纳页：条件信号、残差学习动机、混合数据与适用边界 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 像素域世界模型谱系中的一条「**已知静态几何 + 手轨迹**」分支 |
| [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) | 当把视频扩散当作**交互沙盒**时的代表样本之一（评估型用法见论文 §3.4） |

## 对 wiki 的映射

- 沉淀 **[`wiki/methods/dwm.md`](../../wiki/methods/dwm.md)**；原始论文推导见 [`sources/papers/dwm_arxiv_2512_17907.md`](../papers/dwm_arxiv_2512_17907.md)。
