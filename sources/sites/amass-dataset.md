# AMASS（Archive of Motion Capture as Surface Shapes）

- **标题**: AMASS
- **类型**: dataset / research-portal
- **官网**: https://amass.is.tue.mpg.de/
- **论文**: ICCV 2019 — [AMASS: Archive of Motion Capture as Surface Shapes](http://files.is.tue.mpg.de/black/papers/amass.pdf)（站点摘要页链接）
- **代码 / 教程**: https://github.com/nghorbani/amass
- **机构**: Max Planck Institute for Intelligent Systems, Perceiving Systems（站点页脚）
- **收录日期**: 2026-05-15

## 一句话摘要

将多份**光学标记动捕**数据统一到 **SMPL** 等一致人体参数化表示上的大规模人类运动档案；站点自述适用于动画、可视化与深度学习训练数据生成，下载需注册并遵守站点许可条款。

## 为何值得保留

- **表示统一**：解决「各 MoCap 集骨架/标记集不一致」导致的难以合并训练问题；与 SMPL 生态（重定向、生成模型、Kimodo 等）衔接紧密。
- **规模与多样性**：站点摘要给出约 **40+ 小时**、**300+ 被试**、**11000+ 段动作**量级叙述（以官网与论文为准）。
- **机器人侧常见入口**：在 AMP、ProtoMotions、MimicKit 等管线中常被用作「人体参考运动」的大规模来源之一。

## 站点公开要点（编译自官网摘要）

- 使用 **MoSh++** 将原始标记序列拟合到带蒙皮网格的 **SMPL** 人体模型，并宣称可恢复软组织动态与较自然的手部运动。
- 下载前需在站点 **register**；另有 bib 合并引用、贡献动捕数据联系邮箱等流程说明。

## 对 Wiki 的映射

- **wiki/entities/amass.md**：数据集实体页（归纳级，非论文转存）。
- **wiki/concepts/motion-retargeting.md**：在「工具与数据集」表与参考来源中链接本档案。
