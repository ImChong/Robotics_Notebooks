# 亮源新创发布全身智能基础模型 Light-O1

> 来源归档（blog / 微信公众号）

- **标题：** 亮源新创发布全身智能基础模型 Light-O1
- **类型：** blog
- **作者：** 亮源新创（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/3U3p1ECYP5vdrLqyiiFJdw
- **发表日期：** 2026-09-21
- **入库日期：** 2026-09-21
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **一句话说明：** 亮源新创「规模化预训练」范式首个全身智能基础模型：从互联网视频恢复结构化人类动作、自回归预训练 transferable action prior，验证跨本体 **Transfer Scaling Law**（10 万动作小时）；同步开源 **Light-O1-Preview** 文本→全身动作生成。

## 核心摘录（归纳，非全文）

### 定位：三段范式之「规模化预训练」

- 继 **2026-09-01** [LightNav-0](../papers/lightnav0_arxiv_2608_30935.md)（规模化对齐）、**2026-09-09** [Light REACT](../../wiki/entities/light-react.md)（规模化部署）后，Light-O1 面向 **基础模型预训练** 段。
- 亮源新创三段范式：**规模化预训练 → 规模化对齐 → 规模化部署**；本篇为预训练段首个公开模型成果。

### 问题：机器人动作数据规模瓶颈

- 现有机器人动作模型依赖遥操作、UMI 等 **专项采集**，成本高、多样性难扩展。
- 互联网视频记录人类在不同环境中移动、操作、交互，可在规模与多样性上补充专项数据。

### 机制：人类动作预训练 → 跨本体迁移

| 阶段 | 做法 |
|------|------|
| **数据构建** | 视频策展 → 3D 人类动作恢复 → 细粒度语言标注；统一表示：根轨迹 + 身体姿态 + 手部状态 |
| **预训练** | 语言、视觉、离散动作 token 交错的多模态时序序列；自回归 Transformer（Qwen3.5-4B 基座）预测下一 action token |
| **适配** | 目标机器人遥操作数据 post-training；统一人类动作经 BFM 执行，或 diffusion action expert 映射到目标动作空间 |
| **Scaling Law** | 预训练 token 预算 D 从 3.75B 到 120B（最大 ≈ **10 万动作小时**）；适配 Nymeria / HIW-500 / LightBot 数据后，预测损失与开环 MPJPE 呈 **幂律下降** |

### 能力：Loco-manipulation 与表达性全身技能

- **Loco-manipulation：** 指令定义任务，模型在观测场景中协调移动、姿态与灵巧操作；LightBot 与 Unitree G1 真机演示（递毛巾、擦桌、开鞋柜、捡耳机等）。
- **表达性全身技能：** 指令定义动作本身（单膝求婚、高尔夫挥杆、单腿站立等）；模型先 **语言推理** 再生成 `(frames, 138)` 统一人类动作（20 FPS）。
- **RoboCasa GR-1：** 仿真 24 项厨房桌面任务 macro success **79.3%**（50 episodes/task，tech blog 口径）。

### 开源发布（入库日 2026-09-21 核查）

| 资产 | 状态 |
|------|------|
| Tech Blog | **已上线** <https://www.lightorigins.com/en/blog/light-o1> |
| 代码 | **已开源** <https://github.com/lightorigins/Light-O1>（Apache-2.0） |
| 预览模型 | **已发布** [LightOriginsHQ/Light-O1-Preview](https://huggingface.co/LightOriginsHQ/Light-O1-Preview) |
| Playground | **已上线** [HF Space](https://huggingface.co/spaces/LightOriginsHQ/Light-O1-Preview-playground) |
| 论文 / arXiv | **未列**（tech blog + `@misc` 引用） |
| 完整 Light-O1 训练权重 | **未公开**（仅 Preview 动作生成模型） |

> 以官网 tech blog 与 GitHub README 为准；完整 loco-manipulation 策略权重与预训练 checkpoint 待后续发布。

## 对 wiki 的映射

- 新建 [light-o1.md](../../wiki/entities/light-o1.md) — 全身智能基础模型实体页
- 交叉更新 [light-react.md](../../wiki/entities/light-react.md)、[paper-lightnav-0.md](../../wiki/entities/paper-lightnav-0.md) — 同机构三段范式互链
