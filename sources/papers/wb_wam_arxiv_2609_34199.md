# WB-WAM（异构身–手预训练 · 人形 loco-manipulation WAM）

> 来源归档（ingest）

- **标题：** WB-WAM: Heterogeneous Body-Hand Pre-training for Humanoid Loco-Manipulation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.34199>
- **项目页：** <https://wb-wam.github.io/>
- **机构：** IIIS, Tsinghua University；Xiong'an Institute of Artificial Intelligence；The University of Melbourne
- **作者：** Chuan Qin*、Shaoting Zhu*、Siyuan Luo、Siqiao Huang、Hongyu Zhao、Hang Zhao†（* 共一；† 通讯）
- **代码：** 截至 **2026-09-30** 项目页 **GitHub / Hugging Face 按钮为 disabled**，无公开 URL
- **入库日期：** 2026-09-30
- **一句话说明：** 在生成式视频预训练中注入 **72-D 全身物理动作监督**（body / root / 双手），三阶段 **异构 1880.2 h → PICO 22 h 对齐 mid-training → 3.37 h 真机 post-training**；HumanoidArena **81.9%**，五任务真机均值 **84.0%**；PICO 对齐 mid-training 可用 **30 ep/任务** 机器人数据逼近 **100 ep** 直接适配（**73.8% vs 65.0%**）。

## 核心摘录（MVP）

### 1) 问题：机器人预训练缺「身–手一体」覆盖

- **摘录要点：** 人形 loco-manipulation 需要 **身体运动与灵巧手协同**；现有机器人预训练数据难以覆盖完整全身 motion。异构源常 **只有手标注无身、或只有 motion 无视频**——需统一空间联合学 video + action。
- **对 wiki 的映射：**
  - [WB-WAM](../../wiki/entities/paper-wb-wam.md) — 总览
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 任务语境

### 2) 72-D 共享物理动作空间 + 双 expert WAM

- **摘录要点：** \(\mathbf{a}_t=[\mathbf{q}^B_t,\mathbf{r}_t,\mathbf{q}^L_t,\mathbf{q}^R_t]\in\mathbb{R}^{72}\)：**body 关节参考、root、左右 articulated hand**；各数据源 **部分通道有标注** 即可填入对应坐标。**Video expert + Action expert** 在 vision / language / proprio 条件下 **联合** 预测视觉动力学与全身动作轨迹；部署时 **body/root 经 SONIC 执行**，**hand 直接控 Wuji 指关节**。
- **对 wiki 的映射：**
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md) — 低层全身跟踪
  - [World Action Models](../../wiki/concepts/world-action-models.md)

### 3) 三阶段训练与 WB-Datasets

- **Stage I — 异构预训练：** **1880.2 h**，**9** 个外部源（含 Xperience-10M、MotionMillion、EgoDex 等）；GMR 重定向到 **G1**，人手 → **Wuji** 关节；已有机器人坐标数据做 schema 对齐。
- **Stage II — PICO mid-training：** **22 h**、**73 tasks**、**13,396 episodes**（任务执行段）；PICO 4 Ultra + 五点追踪；SMPL @ 20 Hz → GMR + 约束 WBIK；手 keypoint 用 **MINT** 重建再 retarget Wuji。
- **Stage III — 真机 post-training：** **3.37 h**、**8+2 task groups**、**1,011 episodes**；**SONIC 全身遥操作**（PICO body/root + **MANUS** 手套控 Wuji）；每任务独立适配 + **辅助 FK 监督** body 几何。
- **对 wiki 的映射：**
  - [HumanoidArena](../../wiki/entities/paper-humanoidarena.md) — 仿真评测协议
  - [GMR](../../wiki/methods/motion-retargeting-gmr.md)

### 4) 评测数字（摘要 / 项目页）

- **HumanoidArena（7 任务，SONIC 后端）：** WB-WAM **81.9%** 均值 SR，**七项均超** 报告最强 SONIC 基线。
- **真机五任务：** 无 PICO mid-training 时 WB-WAM **84.0%** 均值 SR vs 最强基线 **OpenWAM 80.0%**（六基线对比）。
- **数据效率：** 四任务有 **对齐 PICO 示范** 时，mid-training + **30** 机器人 demos/任务 → **73.8%**，高于直接 post-training **100** demos → **65.0%**（**−70%** 机端数据）。
- **附加：** 共享 **水果操作** 策略展示 **语言条件** 选目标；未见视觉条件泛化实验（项目页有 visuomotor 叙述）。
- **对 wiki 的映射：**
  - [OpenWAM](../../wiki/entities/paper-openwam.md) — 真机对照基线之一

### 5) 开源状态（步骤 2.5，2026-09-30）

- **摘录要点：** 项目页展示 **arXiv** 链接；**GitHub** 与 **Hugging Face** 资源按钮 **`disabled`**，无 href — 判定 **待发布**（非「确认未开源」：UI 预留发布位）。
- **对 wiki 的映射：**
  - [wb-wam.github.io](../sites/wb-wam-github-io.md)

## 当前提炼状态

- [x] arXiv HTML / API / 项目页对齐
- [x] 步骤 2.5：**待发布**（无 GitHub/HF URL）
- [x] wiki 映射：`wiki/entities/paper-wb-wam.md` 新建
