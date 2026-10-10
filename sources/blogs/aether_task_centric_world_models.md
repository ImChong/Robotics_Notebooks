# Back to Parsimonious Latents: Task-Centric World Models from Visual Foundations（Aether AI 博客）

> 来源归档（blog / Aether AI 官方 Field notes #05）

- **标题：** Back to Parsimonious Latents: Learning Task-Centric World Models from Visual Foundations
- **类型：** blog（论文解读文，配套 arXiv 论文）
- **作者 / 组织：** 页面署名「Paper Fu · Feng · Hansen · Huang」/ Aether AI 博客（aetherlabs.ai）
- **原始链接：** <https://aetherlabs.ai/articles/task-centric-world-models.html>
- **博客索引：** <https://aetherlabs.ai/blog.html>（编号 05 · World Models）
- **发表日期：** 2026-07-09（页面「Published 2026 · 07 · 09」）
- **入库日期：** 2026-10-10
- **抓取方式：** `curl` 抓取静态 HTML 后抽取正文；图表数值只出现在图片中，未转录
- **一句话说明：** 介绍 **TC-WM**：冻结视觉基础模型（DINOv2 等）的嵌入只当「语义脚手架」，用 **一个线性投影** 压到紧凑潜变量 \(z=[z^s,z^c]\)，\(z^s\) 用 InfoNCE 与本体感觉对齐，线性解码器重建嵌入防塌缩，在 \(z\) 上学动力学并做规划；还把同一目标用于微调 16B 视频模型 Cosmos3-Nano（TC-Cosmos3）并在 DROID 真实片段上做开环对比。

## 开源 / 项目页核查（步骤 2.5，截至 2026-10-10）

| 项 | 结论 |
|----|------|
| 论文 | arXiv:2605.25620（见 [论文归档](../papers/tc_wm_arxiv_2605_25620.md)） |
| 项目页 | <https://minghaofu.com/tc-wm/>（可访问，含交互 demo 与 rollout 视频） |
| 代码 | **已开源**：<https://github.com/MinghaoFu/TC-WM>（README 标 MIT；`train.py` / `plan.py` / `rollout.py` 入口，Hydra 配置） |
| 权重 | README 写「will be released」到 HF `MinghaoFu/TC-WM-checkpoints`；2026-10-10 用 HF API 查询返回未授权（**推测** 仓库尚未公开） |
| TC-Cosmos3 | 博客与项目页只给视频对比；仓库 README 未见 Cosmos3 微调脚本说明（encoder 配置项中有 `cosmos_ci`） |

## 核心摘录（归纳，非全文）

### 问题设定

- 世界模型的瓶颈不只是预测更准，而是 **在哪个潜空间里预测**。像素空间（视频生成式模拟器）和冻结基础模型嵌入空间（JEPA、DINO-WM）是两种主流做法；后者会继承编码器的表征偏置——纹理、光照、背景、桌面花纹等对识别有用、但不受动作影响的因素。
- 作者口号：「视觉表征回答场景长什么样；控制需要回答什么对行动重要。」

### 方法要点

- \(x_t\) = 冻结视觉嵌入 ⊕ 可训练的本体感觉嵌入；\(z_t = W x_t\)（**单个线性投影**）；动力学 \(p(z_{t+1}\mid z_{t-H:t}, a_{t-H:t})\)。
- 消融（博客文字描述）：随机投影替换学习到的线性投影 → 性能下降；非线性投影头 → 无提升。作者解读为「更像子空间抽取而非再学一层深表征」。
- \(z=[z^s,z^c]\)：\(z^s\) 与本体感觉 \(s^p\)（关节角、末端位姿）做 InfoNCE 对齐；\(z^c\) 不受约束，承载物体构型、场景等上下文。
- 线性解码器重建原始嵌入，仅训练时使用，作用是防止压缩丢掉预测所需信息。
- 总损失：\(\mathcal L = \mathcal L^z_{dyn} + \mathcal L^s_{dyn} + \lambda_{align}\mathcal L_{align} + \lambda_{rec}\mathcal L_{rec}\)；Transformer 预测下一潜变量，另一头预测下一本体感觉。
- 可辨识性主张：在模型假设下，线性投影把任务中心子空间辨识到 **仿射变换** 为止。
- 完全离线训练，无奖励、无在线交互。测试时：目标到达用 **CEM**；运动类任务博客写「用 SAC 在学到的动力学上训策略」；高维操作用 **潜扩散规划器（LDP）** 提案、世界模型打分。

### 实验（文字结论，具体数值在图中）

- 9 个离线视觉控制环境：导航（Maze、Wall）、操作（Lift、Can、Square、Push-T）、运动（Reacher、Cheetah、Hopper）；基线 TD-MPC2、DreamerV3、MuZero、DINO-WM。
- 潜空间 rollout 误差在「几乎所有任务」最低，图像重建「有竞争力」。
- 规划：饱和的目标到达任务上与最强基线持平；**操作任务上是唯一在每个任务都超过 DINO-WM 的方法**（自报）。
- Lift 上的损失消融：去掉嵌入重建 → 规划与视觉预测都明显变差；去掉本体对齐 → 成功率降、视觉质量基本保持；\(z^s/z^c\) 维度划分影响较小。
- 扰动分析：扰动 \(z^s\) 的响应集中在夹爪与被操作物体并跟随物体移动；扰动 \(z^c\) 的响应分散在背景。

### 扩展到视频世界模型

- 用 TC-WM 目标微调 **Cosmos3-Nano（16B）** 得 TC-Cosmos3，在 DROID 真实片段上以相同起始帧 + 相同真值动作做开环对比（腕部 + 两路外部视角拼接）。
- 博客结论（定性）：TC-Cosmos3 更锚定夹爪与被操作物体，比原始 Cosmos3-Nano「更物理一致、更任务中心」。无定量指标。

## 可信度边界

- 博客与项目页只给图和定性描述，**未给逐任务数值表**；本归档未从图片中读数。
- 规划器口径不一：博客写运动任务用 SAC，项目页与 README 写 Cheetah / Hopper 用 CEM。预测器博客写 Transformer，README 写 ViT。
- TC-Cosmos3 只有定性视频，没有量化评测。

**对 wiki 的映射：** [paper-task-centric-world-models](../../wiki/entities/paper-task-centric-world-models.md)
