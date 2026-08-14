# Green for Go, Red for No: Visual Grounding via Semantic Segmentation for VLA Navigation Policies（arXiv:2607.05122）

> 来源归档（ingest）

- **标题：** Green for Go, Red for No: Visual Grounding via Semantic Segmentation for VLA Navigation Policies
- **类型：** paper / VLA / navigation / visual grounding / semantic segmentation
- **arXiv：** <https://arxiv.org/abs/2607.05122>
- **PDF：** <https://arxiv.org/pdf/2607.05122>
- **HTML：** <https://arxiv.org/html/2607.05122v1>
- **作者：** Adrian Szvoren、Dimitrios Kanoulas、Nilufer Tuptuk
- **机构：** 伦敦大学学院（UCL）计算机系；安全与犯罪科学系
- **资助：** CDT in Cybersecurity（EP/S022503/1）；UKRI FLF RoboHike（MR/V025333/1）
- **入库日期：** 2026-08-14
- **一句话说明：** 对冻结的导航 VLA（OmniVLA）做 **推理时语义分割视觉接地**：SegFormer 把 egocentric 图标成 **绿=可通行 / 红=不可通行**；对照 observation-only 与 joint（观测+目标增强）。Grand Tour ETH-2 上最远航点误差降 **27–44%**，但归一化误差显示收益主要来自 **轨迹变短约 30%**（长度正则），而非单位距离空间推理；**stop** 指令失败不被接地修复。无项目页、**确认未开源**。

## 步骤 2.5：项目页与源码开放核查

- **项目页：** 无（arXiv abs/HTML 未列 `*.github.io` / lab 资源页 / Code 链接）。
- **代码 / 权重 / 数据：** 论文未给出 GitHub / HF / Zenodo；未写 “code will be released”。
- **结论（截至 2026-08-14）：** **确认未开源**。评测依赖已发表的 **OmniVLA**（Hirose et al., arXiv:2509.19480）与 **Grand Tour** 数据集（Frey et al., arXiv:2602.18164）；SegFormer 是通用分割骨干，不是本文管线仓。
- **复现边界：** 可复述方法（绿/红 overlay + 可选语言后缀），但不能从官方仓复现 ETH-2 分段评测脚本。

## 摘要级要点

- **问题：** 导航 VLA 仍易受感知干扰与场景歧义；视觉接地在 VLM/VQA（Set-of-Marks、PIVOT）与操作 VLA（BYOVLA 遮挡无关区）上有效，但 **导航 VLA 上缺少实证**。
- **约束：** 实时推理、未见环境泛化（不绑固定物体词表）、多模态目标（语言 + 目标图像）。
- **方法：** SegFormer 二值可通行分割 → 绿/红 overlay 得到 \(i'\)。
  - observation-only：\(\tau_1 = VLA(i', g)\)
  - joint：语言目标追加 *keep the trajectory within green traversable areas and avoid red non-traversable areas*；图像目标同样 overlay → \(\tau_2 = VLA(i', g')\)
- **为何不用 SoM / YOLO：** SoM 链式提示把单图从 ~0.5 s 拉到 ~5 s；YOLO 固定类别，限制未见障碍泛化。
- **评测：** 冻结 **OmniVLA**（7 航点 2D 轨迹）；**Grand Tour ETH-2** 室内段（ANYmal D）；NVIDIA Jetson Thor。三配置：omnivla-base 语言、omnivla-finetuned-cast 语言、omnivla-base 图像目标。楼梯段与 **stop** 段在总表中剔除（楼梯被标成不可通行；stop 为训练缺口）。
- **主结果（语言 / omnivla-base）：** 最远航点 WP7 平均误差 **0.22 m → 0.16 m（seg, −27%）→ 0.15 m（aug, −32%）**；长指令 WP7 **−39–44%**，短指令 **−35–36%**。图像目标基线已 **0.13 m**，接地几乎无增益（0.12–0.11 m）。
- **机制读法：** 接地轨迹长度 **0.63 m → 0.45 / 0.43 m（约 −30%）**；按长度归一化后优势消失，WP1–5 基线归一化误差甚至略低 → **主要是轨迹长度正则，不是单位距离空间推理提升**。
- **失败模式：** 语言/图像「stop」从不输出零位移，始终向前走；接地三种条件都修不掉 → 感知增强补不了缺失训练信号。
- **joint vs obs-only：** 约一半片段 joint 不优于 obs-only；增强主要帮含糊指令，描述性指令上后缀冗余。

## 核心摘录（面向 wiki 编译）

### 问题形式

\[
\tau = VLA(i, g),\quad g\in\{g_l, g_i\},\quad \tau=\{(x_k,y_k)\}_{k=1}^{7}
\]

### 关键数字（索引级，以原文 Table I / 正文为准）

| 设定 | Base | Segmented | Augmented |
|------|------|-----------|-----------|
| omnivla-base 语言 WP7 | 0.22 m | 0.16 m（−27%） | 0.15 m（−32%） |
| 短指令（2–7 词）WP7 | 0.187 m | 0.120 m | 0.122 m |
| 长指令（≥8 词）WP7 | 0.160 m | 0.090 m | 0.098 m |
| 去 stop 后 1–7 航点均值 | 0.095 m | 0.064 m | 0.064 m |
| 预测轨迹平均长度 | 0.63 m | 0.45 m | 0.43 m |
| 「stop」WP1 | 0.204 m | 0.149 m | 0.100 m（理想为 0） |

### 与相邻工作的分界

| 路线 | 与本文的分界 |
|------|----------------|
| **Set-of-Marks / PIVOT** | VLM 视觉提示；延迟不适合连续导航 |
| **BYOVLA** | 操作 VLA 遮挡任务无关区；本文标 **可通行性** 而非物体相关性 |
| **NaVILA** | 训中层语言动作 + 腿式 locomotion；本文 **不重训**，只改输入 |
| **OmniVLA** | 被评测的冻结策略；本文不是新 VLA |
| **NavWAM 等 image-goal** | 本文显示图像目标上 grounding 增益很小（视觉目标已提供空间线索） |
| **Green-VLA（Sber）** | **同名易混**：那是分阶段通才操作 VLA，与本绿/红 overlay **无关** |

## 对 wiki 的映射

- 沉淀实体页：[Green for Go · VLA 导航可通行性视觉接地](../../wiki/entities/paper-green-for-go-vla-nav-grounding.md)
- 交叉补强：[视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)、[VLA](../../wiki/methods/vla.md)、[NaVILA](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)、[NavWAM](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md)、[ActFovea](../../wiki/entities/paper-actfovea.md)（同为冻结 VLA 上的推理时干预）、[Green-VLA](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md)（消歧）

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2607.05122>
- OmniVLA：<https://arxiv.org/abs/2509.19480>
- Grand Tour：<https://arxiv.org/abs/2602.18164>
- SegFormer：Xie et al., NeurIPS 2021
