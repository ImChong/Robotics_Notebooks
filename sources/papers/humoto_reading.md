# HUMOTO: A 4D Dataset of Mocap Human Object Interactions

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记 + arXiv v2 全文通读）

- **标题：** HUMOTO: A 4D Dataset of Mocap Human Object Interactions
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2504.10414>（v1: 2025-04-14，v2: 2025-10-15；ICCV 2025）
- **项目主页：** <https://jiaxin-lu.github.io/humoto/> · <https://adobe-research.github.io/humoto/>
- **官方代码：** <https://github.com/Jiaxin-Lu/humoto>
- **核对范围：** 全文核对 arXiv 文本 §3–§6（场景驱动 LLM 编剧管线、Rokoko + 双 Kinect + FoundationPose + SAM2 抗遮挡采集、两阶段美术清洗 QA 与多级文本标注）、Table 1–2（物理与生成质量评测及 26 人用户研究）、Fig. 11 数据统计与附录 A（PyBullet CoACD、MotionGPT / 4D Humans / TRAM / Affordance Diffusion 验证）。
- **通读：** 已完成
- **入库日期：** 2026-07-10（原文通读复核：2026-11-14）
- **一句话说明：** 针对人-物交互（HOI）数据「遮挡严重、缺乏任务目的、脚滑穿模」的三大痛点，HUMOTO 结合 LLM 场景剧本生成、Rokoko 动捕服/EMF 手套 + 双 Kinect 多视物体跟踪以及两阶段专业美术清洗，构建了 735 段（Fig. 11 统计 736 段）、约 7,875 秒（30 fps）、覆盖 63 件精密物体（72 个功能部件）与 Mixamo 兼容骨架的 4D HOI 数据集（ICCV 2025）。

## 核心摘录（策展与核对摘要）

- **规模与口径核对**：摘要及正文报告 735 段、7,875 s（30 fps，约 2.2 小时）、63 件物体、72 个部件、单场景最多 15 个物体、单表演者；Fig. 11 直方图显示 736 段 / 7874.4 s（131.2 min）、单段时长 3.0–32.0 s。
- **物理与交互质量（Table 1）**：Foot Sliding 0.958 cm、Jerk 1.87、MSNR 9.42 dB、Coherence 0.653、Diversity 0.956、Object Jerk 1.13、Penetration 0.0068 cm、Contact Entropy 1.4587、State Consistency 0.5061。
- **下游验证**：通过 PyBullet CoACD 接触物理校验、26 名受试者主观研究以及 MotionGPT / 4D Humans / TRAM / Affordance Diffusion 基准评测。

## 对 wiki 的映射

- [paper-notebook-humoto-a-4d-dataset-of-mocap-human-object-intera](../../wiki/entities/paper-notebook-humoto-a-4d-dataset-of-mocap-human-object-intera.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions.html>
- 论文：<https://arxiv.org/abs/2504.10414>
