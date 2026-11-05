# Dissecting and Open-Sourcing Multitask Diffusion Transformer Policy（Bryson Jones）

> 来源归档（ingest）

- **标题：** Dissecting and Open-Sourcing Multitask Diffusion Transformer Policy
- **作者：** Bryson Jones
- **类型：** blog（Substack，页面日期 2026-03-30；LeRobot 文档 BibTeX 标注年份 2025，两者不一致，本页以抓取到的页面日期为准）
- **URL：** <https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-diffusion-transformer-policy>
- **入库日期：** 2026-11-05
- **相关代码：** 独立仓库 `brysonjones/multitask_dit_policy`（文中链接，本次未读）；LeRobot 集成版 `src/lerobot/policies/multi_task_dit/`
- **一句话说明：** 作者依据公开的 TRI LBM 论文与自己的摸索复现 Multitask DiT Policy，并同时开源为独立仓库与 LeRobot 集成版。

## 核心摘录（面向 wiki 编译，读自页面文本）

- **动机：** 文中称 Boston Dynamics 的 Atlas 演示里用于轨迹生成的是 Multitask Diffusion Transformer Policy，且这不是 VLA（没有预训练 VLM 骨干，用的是小得多的视觉与文本编码器）；因当时没有公开代码，作者根据已公开的 TRI 文献自行实现。
- **作者自述的使用经验（未核实）：** 在单任务到少任务、约 100 至 200 条示范的数据集上，单张 H200 训练约 10 小时即可得到不错的结果。
- **视觉编码器：** CLIP ViT-B/16，只取 CLS token，端到端联合训练，学习率为其余参数的 1/10；另提供 DINOv3 编码器选项，作者称效果未见更好。
- **文本编码器：** CLIP 文本 tokenizer 与嵌入层冻结，只训练一个投影层；作者承认语言条件因此偏弱。
- **本体感知：** 关节角等不经编码直接拼进条件向量。
- **主干：** 将 Meta 的 DiT 从 2D 图像生成改为 1D 序列建模；视觉、任务文本、本体感知与扩散时间步特征拼成单一条件向量；位置编码默认用 RoPE，作者称相比绝对位置编码有稳定提升。
- **目标函数：** 同时实现经典扩散（DDPM/DDIM）与 flow matching。
- **LeRobot 集成：** 作者称集成版"尽量贴近朴素实现"，便于上手；独立仓库含更多实验性编码器与模块化扩展。

## 对 wiki 的映射

- [multitask-dit-policy](../../wiki/entities/multitask-dit-policy.md)
- 对应的论文页（不同节点）：[LBM 论文条目](../../wiki/entities/paper-rcl-2507-05331-a-careful-examination-of-large-behavior-models-f.md)（arXiv:2507.05331）

## 当前提炼状态

- [x] 博文正文前半（架构、编码器、目标函数部分）已读；后半（训练细节、失败模式）仅略读未核对
- [ ] 独立仓库 `multitask_dit_policy` 未读
