# Awesome Egocentric Vision README 通读（技术地图视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/sun-awesome-ego-technology-map.md`）

- **标题：** Awesome Egocentric Vision（README.md，英文）
- **作者 / 维护方：** Yuanliang Sun（sun254667）
- **类型：** repo / awesome-list / curated-index
- **原文：** <https://raw.githubusercontent.com/sun254667/awesome-egocentric-vision/main/README.md>（2026-11 抓取，1698 行；CC0-1.0）
- **一句话说明：** 第一人称视觉与具身 AI 的精选清单，按 10 个研究主题 + 数据集 + 智能眼镜 + 挑战表组织，每条给出出处（会议 / arXiv 年份）、亮点一句话和论文链接。
- **核对范围：** 通读 README 全文结构与每个章节：§1 综述（5 条）、§2.1–2.10 全部条目的标题 / 出处 / 亮点、各节数据集表、§3 数据集与仿真器、§4 智能眼镜、§5 相关清单、§6 关键研究挑战（含总体挑战 4 条与分方向挑战表）。阅读对象是清单本身；清单所链接的 200+ 篇论文 / 数据集主页未逐篇打开，论文内部结论不在本次核对内。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **收录标准：** 优先 CVPR / ICCV / ECCV / NeurIPS / ICLR / AAAI / TPAMI / TVCG 等顶会顶刊；重点方向为 Ego-VLM/LLM、Ego-VLA、3D/4D 重建、手物交互、注视估计、长视频理解、世界模型。
- **结构：** 综述 → 10 个核心主题（2.1 动作识别/检测/预期、2.2 手物交互、2.3 Ego-VLM 与多模态推理、2.4 长视频、2.5 注视与意图、2.6 3D/4D 重建、2.7 Ego-VLA、2.8 Ego 世界模型、2.9 程序化 AI 助手、2.10 数据安全与隐私）→ 数据集/基准/仿真器 → 智能眼镜 → 相关清单 → 挑战。
- **条目格式：** 每条含 Publication、Highlights、Paper Link；多数主题末尾附「Related Datasets & Benchmarks」表（数据集名 / 年份 / 规模 / 任务 / 链接），⭐ 标记锚点数据集（如 Ego4D、EPIC-KITCHENS-100、Ego-Exo4D、EgoDex、HOI4D、Assembly101）。
- **内容时间分布：** 清单以 2026 年 arXiv 条目为主，同时保留 2009–2023 的经典数据集（如 Handled Objects 2009、ADL 2012、EGTEA Gaze+ 2018）。
- **总体挑战（§6）：** 视野窄 / 自遮挡 / 抖动模糊；数据稀缺与隐私；人与人之间的分布偏移；全天候超长序列建模的记忆与长程依赖瓶颈。
- **分方向挑战（§6 表）：** 动作识别的动宾组合歧义；HOI 的接触状态标注成本与相互遮挡；Ego-VLM 缺第一人称常识与时空多步推理；长视频的上下文窗口与场景图构建；注视的个体差异；单目深度歧义与动态场景；Ego-VLA 的人机具身差距与动作空间错位；世界模型长 rollout 的时序一致性；程序化助手的错误边界模糊与主动干预时机；隐私中旁观者与佩戴者的平衡、GDPR。
- **Ego-VLA 要点（§2.7 与挑战表）：** 清单点名 HumanScale（人类第一人称视频预训练优于真机数据，清单写 +52.5% 成功率）、EgoScale（2 万+ 小时带动作标注数据训练 VLA）、ACE-Ego-0、EgoWAM（DINO 目标提升 OOD 泛化、3D flow 提升域内）等；以上数字均为清单转述，未回到原论文核对。
- **智能眼镜（§4）：** 基准与数据（SUPERGLASSES、Project Aria 数据集）、低功耗端侧感知（OpenGlass、EgoTrigger）、终身记忆助手（EGAgent、LightMem-Ego）、人机交互（VisionClaw 等）、数据采集基础设施（AoE）。
- **隐私（§2.10）：** EgoPrivacy、EgoNormia、EAPrivacy 三个基准，EgoBlur / Ego4D 隐私框架，及欧盟 GDPR 对智能眼镜的监管报道。

## 对 wiki 的映射

- [`wiki/overview/sun-awesome-ego-technology-map.md`](../../wiki/overview/sun-awesome-ego-technology-map.md)
- [`wiki/entities/awesome-egocentric-vision.md`](../../wiki/entities/awesome-egocentric-vision.md)
- 清单级结构摘要：[`sources/repos/awesome-egocentric-vision.md`](../repos/awesome-egocentric-vision.md)
