# AdvDex: Learning Dexterous Manipulation from Human Demonstrations via Joint-Aligned Actions and Adversarial Learning（arXiv:2608.14028）

> 来源归档（ingest）

- **标题：** AdvDex: Learning Dexterous Manipulation from Human Demonstrations via Joint-Aligned Actions and Adversarial Learning
- **缩写 / 框架：** **AdvDex** / **JAAS**（Joint-Aligned Action Space）/ **OmniShare**
- **类型：** paper / vla / dexterity / cross-embodiment / human-demonstration
- **arXiv：** <https://arxiv.org/abs/2608.14028>（PDF：<https://arxiv.org/pdf/2608.14028>）
- **项目页：** 论文与摘要未列独立项目页（截至入库日）
- **作者：** Zhiyue Zhao、Jingyi Wu、Hairuo Liu、Mingyu Liu、Liyang Li、Hengdi Zhang、Tong He、Zhengxue Cheng（通讯）
- **机构：** 浙江大学（ZJU）；复旦大学（Fudan）；上海创智学院（Shanghai Innovation Institute）；上海交通大学（SJTU）；帕西尼（Paxini Tech）
- **入库日期：** 2026-08-17
- **一句话说明：** 用 OmniShare 人手数据 + SE(3) 腕与 15 指关节规范动作空间 JAAS + 域对抗视觉，把人手/灵巧手/夹爪接到同一 VLA。

## 开源状态（步骤 2.5）

- **项目页：** 无独立 `*.github.io` / lab 页可核。
- **论文：** 未给 GitHub / HF / 数据集下载链；亦未写 “code will be released”。
- **结论：** **确认未开源**（代码、权重、OmniShare 均未公开）。

## 摘录 1：问题与数据（§1、§3.1）

- **痛点：** 真机灵巧演示贵；Wuji / XHand / Shadow / DexH13 等动作空间碎片化；共享视觉编码器会把任务几何与本体外观缠在一起。
- **OmniShare：** 宣称 >100k 轨迹、500+ 任务、700+ 物体；表 1 写 168k traj / 14 视角 / 721 物体 / Dense 文本。采集：微秒同步数据手套（29 磁编 + 霍尔触觉）+ 多视角。
- **处理：** ArUco / FoundationPose 估腕与物体；物理感知优化重定向到 MANO；距离衰减保留接触时序。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-advdex.md`](../../wiki/entities/paper-advdex.md)；与 [UHAS](../../wiki/methods/uhas-unified-hand-action-space.md) 对照「统一的是策略动作还是球面形变」。

## 摘录 2：JAAS 与对抗学习（§3.2）

- **JAAS：** 共享 \(\mathrm{SE}(3)\) 腕 + 15 指关节（每指 3-DoF 欧拉）。MANO 51-DoF、19-DoF 灵巧手、7-DoF 臂+夹爪按功能槽对齐；缺失槽 mask 损失。
- **架构：** VLM 出 cognition token \(z_t\) → DiT 在 JAAS 上扩散去噪动作 chunk；域判别器经 GRL 预测 human / dexterous / gripper。
- **损失：** \(\mathcal{L}_{\mathrm{final}}=\mathcal{L}_{\mathrm{MSE}}+\lambda\mathcal{L}_{\mathrm{D}}\)。推理时去掉对抗支路。

**对 wiki 的映射：** 写清「动作对齐 ≠ 视觉不变」必须同时做。

## 摘录 3：评测（§4）

- **预训练配比：** OmniShare : VITRA-1M : OXE = 5:4:1；再在目标手上少量后训练。
- **手动作预测（mm）：** OmniShare-Unseen \(d_{\mathrm{h-o}}\) 3.2 / MPJPE 2.8 / MWTE 2.5（VITRA 20.1 / 16.2 / 14.8）。
- **真机 Paxini Tora + DexH13：** 五任务 seen 上优于 \(\pi_{0.5}\) / VITRA；未见物体 50%、未见环境 60%。
- **零样本人→机共训：** 人任务与机任务互斥；Box Doll 60%、Press Button 70%、Move Bottle 45%、Tool Use 30%。
- **少样本：** 单物体抓取 0-shot 非零，5 条演示即明显抬升；去对抗比去 OmniShare 掉得更狠。
- **局限：** 精细技能仍受硬件差限制；只评一台灵巧平台；JAAS 不显式建模动力学/接触约束。

**对 wiki 的映射：** 强调「统一动作空间 + 对抗去本体捷径」对少样本/零样本的贡献，并点明未开源。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-advdex.md`**；注册机构 **paxini**。
- 交叉更新 UHAS、VLA、跨具身迁移选型、操作任务页。
