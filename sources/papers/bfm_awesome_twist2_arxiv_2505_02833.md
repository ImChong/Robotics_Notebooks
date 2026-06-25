# TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System

> 来源归档（ingest · awesome-bfm-papers 第 10/41）

- **标题：** TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System
- **类型：** paper
- **BFM 分类：** 02 Goal-conditioned 学习（[awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)）
- **出处：** 2025 · arXiv · **ICRA 2026 接收**（项目页）
- **论文链接：** <https://arxiv.org/abs/2511.02832>
- **项目页：** <https://yanjieze.com/projects/TWIST2/>
- **代码/数据：** <https://github.com/amazon-far/TWIST2> · <https://twist-data.github.io/>
- **机构：** Amazon FAR；Stanford；USC；UC Berkeley；CMU
- **索引来源：** [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) · [具身智能研究室 BFM 41 篇编译](../blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)
- **入库日期：** 2026-05-26（2026-06-12 增补项目页一手摘录）
- **一句话说明：** 便携可扩展全身人形数据采集：2-DoF 颈 egocentric 感知 + PICO 全身遥操作 + 分层 visuomotor（RL 跟踪 + 扩散模仿）；全栈开源与社区数据集。

## 核心摘录（项目页 + 策展）

### 1) 系统四支柱

- **链接：** <https://yanjieze.com/projects/TWIST2/>
- **摘录要点：** **Portability**（任意环境快速部署）、**Holisticness**（egocentric + 灵巧 + 全身）、**Scalability**（高效规模化采集）、**Autonomy**（从数据学 visuomotor 全身控制）。
- **对 wiki 的映射：** [TWIST2](../../wiki/entities/paper-twist2.md)

### 2) 硬件：颈增广 + PICO 便携遥操作

- **摘录要点：** 自研 **2-DoF 颈** 提供主动 egocentric 感知（MuJoCo 建模）；**PICO 4 Ultra + 2 Motion Trackers**；**XRoboToolkit** 统一 egocentric 视频流与全身姿态流（Passthrough / Egocentric 模式）。
- **对 wiki 的映射：** [teleoperation](../../wiki/tasks/teleoperation.md)

### 3) 分层策略学习

- **摘录要点：** **System 1**：sim2real RL **低层全身 tracking**；**System 2**：从 TWIST2 数据 **模仿学习** 高层 visuomotor（如 **Diffusion Policy** 预测未来全身关节位置，含上下身 ghost 轨迹）。
- **对 wiki 的映射：** [diffusion-policy](../../wiki/methods/diffusion-policy.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)

### 4) 采集效率与长时程技能

- **摘录要点：** 15 min **128** 次双手灵巧 pick-place；15 min **50** 次移动 pick-place；长时程叠毛巾、找布折叠、踢球、穿门搬运、地面捡砖等；自主 **Kick-T**、**WB-Dex**。
- **对 wiki 的映射：** [TWIST2](../../wiki/entities/paper-twist2.md)

### 5) 开源与谱系

- **摘录要点：** 训练/部署代码、控制器 checkpoint、颈 BOM/3D 打印全开源；数据集 twist-data.github.io；建立在 **TWIST**（跟踪控制器）、**GMR**（重定向）、**iDP3**（3D 扩散 visuomotor）之上。
- **对 wiki 的映射：** [paper-twist](../../wiki/entities/paper-twist.md)、[motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)

## 对 wiki 的映射

- [TWIST2（论文实体）](../../wiki/entities/paper-twist2.md)
- [teleoperation](../../wiki/tasks/teleoperation.md)
- [behavior-foundation-model](../../wiki/concepts/behavior-foundation-model.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2511.02832>
- 项目页：[twist2-project.md](../sites/twist2-project.md)
- 代码：[twist2.md](../repos/twist2.md)
- 策展列表：<https://github.com/friedrichyuan/awesome-bfm-papers>
