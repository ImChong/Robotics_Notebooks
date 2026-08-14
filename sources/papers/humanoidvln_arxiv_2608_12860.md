# HumanoidVLN（arXiv:2608.12860）

> 来源归档（ingest）

- **标题：** HumanoidVLN: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments
- **类型：** paper / VLN / humanoid / Isaac Sim / benchmark / sim2real
- **arXiv：** <https://arxiv.org/abs/2608.12860>
- **PDF：** <https://arxiv.org/pdf/2608.12860>
- **HTML：** <https://arxiv.org/html/2608.12860>
- **项目页：** <https://humanoid-vln.github.io/>
- **作者：** Quan-Dung Pham, Anh Dao, The-Anh Nguyen, Minh Nguyen-Dinh, Phuong Nam Dang, Tri Pham, Hung Tran, Bach Dao, Tuyen P. Le, Truong Nguyen（VinMotion）；Quan Nguyen（南加州大学，USC）
- **机构：** 越南人形机器人（VinMotion, Inc.）；南加州大学（USC）
- **入库日期：** 2026-08-14
- **一句话说明：** 在 NVIDIA Isaac Sim 上做人形专属物理 VLN：四本体分层控制（RL 步态 + PD/MPC 跟踪）、≥100 m² 可通行场景、Generator–Reviewer–Paraphraser 指令 + 人工核验；零样本评测 NaVILA / DualVLN / StreamVLN / JanusVLN；G1 上 DualVLN 20 条 sim–real 试点。代码宣称录用后开源。

## 摘要级要点

- **问题：** 既有 VLN 基准用运动学传送或把人形当「多种机器人之一」共用控制代理；双足步态约束、跨本体形态差、行走引起的相机抖动都不进评测。
- **平台：** Isaac Sim；四本体（Unitree G1 / H1、Internal-A / Internal-B）下身 10–12 DoF、身高 1.17–1.80 m；低层 **RL locomotion** 出关节力矩，高层 **PD**（离散动作模型）或 **MPC**（连续速度模型）跟踪全局计划。
- **场景：** 87 个室内场景（17 类 / 6 域），可通行面积 ≥100 m²（中位 266、均值 387 m²）；来源为艺术家场景（GRScenes 等）与 **3DGS Real2Sim**（gsplat + 无偏深度 + 深度–法向一致 + TSDF → USDZ）。
- **数据：** 933 条碰撞感知参考 episode；每条 1 条细粒度指令 + 3 条粗粒度风格变体（Formal / Natural / Casual）。**仅作零样本评测集**，不用于训练被评模型。
- **指令管线（MAA）：** 双生成器（Gemma-4-31B-it、InternVL3.5-38B）只看 egocentric 关键帧建路线图；Qwen3-VL-30B-A3B 做目标接地与几何核验；GPT-5.5 改写风格且须保持动作/地标顺序；三人人工终审，20% 双标。
- **评测：** JanusVLN 四本体平均 **SR 43.55% / nDTW 48.38**；H1 平均 SR 仅 20.84%，NaVILA/StreamVLN 在 H1 上 Fall Rate **70.95% / 64.52%**。
- **Sim2Real 试点：** DualVLN + Unitree G1，两场景各 10 条（N=20）；仿真 vs 真机 NE Pearson **r=0.935**，绝对差均值 **0.68 m**，轨迹 nDTW **0.782±0.188**。
- **开源（截至 2026-08-14）：** 论文与项目页写 *Code, benchmark, and data will be released upon acceptance*；项目页 **未列 GitHub / Hugging Face** → **宣称将开源 / 待发布**。

## 核心摘录（面向 wiki 编译）

### 三道缺口与对应解

| 缺口 | 既有做法 | HumanoidVLN |
|------|----------|-------------|
| 仿真 | Habitat 运动学步进；VLN-PE / VLNVerse 把人形当多种本体之一 | 每本体独立 RL 步态 + 可换 PD/MPC 跟踪器 |
| 场景 | 扩场景数、不筛可通行 | 只收 ≥100 m² 可通行地面；3DGS 可免全艺术家资产 |
| 指令 | 单次 VLM 易左右/时序幻觉；全人工太贵 | 结构化路线图对账 + 几何先验核验 + 人工终审 |

### 本体表（论文 Table II）

| 机器人 | 下身 DoF | 身高 (m) | 相机高度 (m) |
|--------|----------|----------|--------------|
| Unitree G1 | 12 | 1.32 | 1.25 |
| Unitree H1 | 10 | 1.80 | 1.72 |
| Internal-A | 12 | 1.61 | 1.54 |
| Internal-B | 12 | 1.17 | 1.11 |

Internal 平台细节因双盲审查省略；不要臆测为某款消费级机型。

### 零样本结果读法（Table III，n=933）

- **模型排序（四本体平均 SR）：** JanusVLN 43.55% > DualVLN ≈ NaVILA > StreamVLN 23.63%。
- **停止质量：** NaVILA 平均 OS–SR 差 **17.39** 点，停靠不可靠。
- **路径保真：** DualVLN 平均 nDTW 43.39（除 Janus 外最高），连续速度 + MPC 更贴参考路径。
- **摔倒：** DualVLN 跨本体 FR 最低；H1 上离散模型 FR 暴涨。SR 不能单独当物理可执行性。

### Sim–real 试点边界

两场景、一个 DualVLN checkpoint、G1 + RealSense D435i；Jetson AGX Orin 传感控制、RTX A6000 Pro 推理。作者明确：**不能**外推到场景级泛化，只证明这对重建场景在 episode 级难度与轨迹结构上与真机一致。

## 对 wiki 的映射

- 沉淀实体页：[HumanoidVLN](../../wiki/entities/paper-humanoidvln.md)
- 交叉补强：[视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)、[VLN 分类 01 数据集与仿真平台](../../wiki/overview/vln-category-01-datasets-platforms.md)、[VLN 四范式开源复现](../../wiki/overview/vln-open-source-repro-paradigms.md)、[NaVILA](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)、[Isaac Sim](../../wiki/entities/isaac-sim.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[VLN-CE](../../wiki/entities/paper-vln-02-vln-ce.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2608.12860>
- 项目页核查：[humanoid-vln-github-io.md](../sites/humanoid-vln-github-io.md)
