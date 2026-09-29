# Robbyant GitHub / Hugging Face 组织（github.com/robbyant）

> 来源归档（ingest · 官方代码与权重组织页）

- **标题：** Robbyant — "Intelligence in Action, Benefits for Everyone."
- **类型：** repo-org / 官方 GitHub organization + Hugging Face organization
- **机构：** 蚂蚁灵波科技（Robbyant），蚂蚁集团（Ant Group）旗下
- **GitHub：** <https://github.com/robbyant>（组织主页写官网 <https://technology.robbyant.com/>）
- **Hugging Face：** <https://huggingface.co/robbyant>
- **ModelScope：** <https://www.modelscope.cn/organization/Robbyant>
- **入库日期：** 2026-09-29
- **一句话说明：** LingBot 系列的一手代码与权重入口：9 个公开模型仓 + `.github` 配置仓，覆盖 Vision / Depth / Map / Video / World / VA / VLA。
- **为什么值得保留：** 各 LingBot 项目的开源状态（代码、权重、许可证）都以此处实际链接为准；公众号清单与媒体通稿里常见失效的 `antgroup/lingbot` 旧链接，需回到本组织核对。

## 公开仓库（2026-09-29 抓取，仓库描述为官方原文）

| 仓库 | 官方描述 | 论文 | 许可证 |
|------|----------|------|--------|
| [lingbot-vision](https://github.com/robbyant/lingbot-vision) | Self-supervised learning for spatial perception | [arXiv:2607.05247](https://arxiv.org/abs/2607.05247) | Apache-2.0 |
| [lingbot-depth](https://github.com/robbyant/lingbot-depth) | Masked Depth Modeling for Spatial Perception | [arXiv:2601.17895](https://arxiv.org/abs/2601.17895) | Apache-2.0 |
| [lingbot-map](https://github.com/robbyant/lingbot-map) | (ECCV 2026 oral & best paper candidate) LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction | [arXiv:2604.14141](https://arxiv.org/abs/2604.14141) | Apache-2.0 |
| [lingbot-video](https://github.com/robbyant/lingbot-video) | Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence | [arXiv:2607.07675](https://arxiv.org/abs/2607.07675) | Apache-2.0 |
| [lingbot-world](https://github.com/robbyant/lingbot-world) | Advancing Open-source World Models | [arXiv:2601.20540](https://arxiv.org/abs/2601.20540) | Apache-2.0 |
| [lingbot-world-v2](https://github.com/robbyant/lingbot-world-v2) | Infinite Worlds with Versatile Interactions | [arXiv:2607.07534](https://arxiv.org/abs/2607.07534) | CC BY-NC-SA 4.0 |
| [lingbot-va](https://github.com/robbyant/lingbot-va) | [RSS 2026] Causal video-action world model for generalist robot control | [arXiv:2601.21998](https://arxiv.org/abs/2601.21998)；仓内另有 `LingBot_VA2_paper.pdf`（VA 2.0 技术报告） | Apache-2.0 |
| [lingbot-vla](https://github.com/robbyant/lingbot-vla) | A Pragmatic VLA Foundation Model | [arXiv:2601.18692](https://arxiv.org/abs/2601.18692) | 以 README 为准 |
| [lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2) | From Foundation to Application | [arXiv:2607.06403](https://arxiv.org/abs/2607.06403) | 以 README 为准 |

## 主要技术信息（README 摘录）

- **LingBot-Vision：** 自监督 ViT 骨干家族，ViT-S/16 到 1.1B 参数 ViT-g/16；旗舰目标为 **masked boundary modeling**（以边界为中心的掩码建模），特征同时包含语义分组与几何结构。
- **LingBot-Depth：** 输入 RGB + 原始深度 + 相机内参，输出精修深度与相机系点云；训练集共 3,019,200 样本（RobbyReal 1.4M 真实室内、RobbyVla 581K 机器人操作、RobbySim 1M 仿真）。
- **LingBot-Map：** 前馈流式 3D 基础模型，输出相机位姿、深度与点云；README 自报 518×378 分辨率约 20 FPS、>10,000 帧稳定推理。
- **LingBot-Video：** 称为首个面向具身智能的开源大规模 MoE 视频生成模型；Dense 1.3B 与 MoE 30B-A3B（+ refiner）两档；"massive web videos integrated with 70,000+ hours of embodied data"。
- **LingBot-World：** 源自视频生成的开源世界模拟器；Base (Cam) 相机位姿控制、Base (Act) 动作控制、Fast（KV cache）三种变体。
- **LingBot-World 2.0：** 又名 LingBot-World-Infinity；无界交互视界、720p 60 fps；pilot agent 规划角色行为、director agent 合成新环境要素；14B / 1.3B 变体。
- **LingBot-VA：** 自回归视频–动作世界建模，dual-stream Mixture-of-Transformers（MoT）+ 异步执行 + KV cache；README 自报 RoboTwin 2.0 Easy/Hard 92.9% / 91.6%、LIBERO 平均 98.5%。

## Hugging Face 权重（2026-09-29，`/api/models?author=robbyant`）

- Vision：`lingbot-vision-vit-{small,base,large,giant}`
- Depth：`lingbot-depth`、`lingbot-depth-pretrain-vitl-14`、`lingbot-depth-pretrain-vitl-14-v0.5`、`lingbot-depth-postrain-dc-vitl14`
- Map：`lingbot-map`
- Video：`lingbot-video-dense-1.3b`、`lingbot-video-moe-30b-a3b`、`lingbot-video-moe-dmd-30b-a3b`、`lingbot-video-rewriter-lora`
- World：`lingbot-world-base-cam`、`lingbot-world-base-act-preview`、`lingbot-world-fast`（及 diffusers 版）；World 2.0：`lingbot-world-v2-14b-{causal-fast,causal-pretrain,bid}`、`lingbot-world-v2-1.3b-causal-fast`
- VA：`lingbot-va-base`、`lingbot-va-posttrain-robotwin`、`lingbot-va-posttrain-libero-long`
- VLA：`lingbot-vla-4b`、`lingbot-vla-4b-depth` 及 RoboTwin 后训练版；VLA 2.0：`lingbot-vla-v2-6b`、`lingbot-vla-v2-6b-robotwin`

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| Vision / Depth / Map / Video / World / World 2.0 / VA 1.0 / VLA 1.0 / VLA 2.0 | **已开源**：代码仓 + HF 权重均可访问 |
| **VA 2.0** | **部分公开**：技术报告 PDF 在 `lingbot-va` 仓；截至 2026-09-29 HF 组织未见 VA 2.0 专属权重（仅 `lingbot-va-base` 等 1.0 系列），**权重未确认开源** |

## 对 wiki 的映射

- [Robbyant（蚂蚁灵波）公司实体](../../wiki/entities/robbyant.md)
- [LingBot-Vision](../../wiki/entities/cn-os-lingbot-vision.md) · [LingBot-Depth](../../wiki/entities/cn-os-lingbot-depth.md) · [LingBot-Video](../../wiki/entities/cn-os-lingbot-video.md)
- [LingBot-Map](../../wiki/methods/lingbot-map.md) · [LingBot-World](../../wiki/entities/lingbot-world.md) · [LingBot-World 2.0](../../wiki/entities/paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling.md)
- [LingBot-VA](../../wiki/entities/paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g.md) · [LingBot-VLA](../../wiki/entities/lingbot-vla.md) · [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md)
