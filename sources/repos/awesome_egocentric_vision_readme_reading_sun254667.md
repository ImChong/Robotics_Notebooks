# Awesome Egocentric Vision README 通读（sun254667）

> 来源归档（原文通读 · 支撑 `wiki/entities/awesome-egocentric-vision.md`）

- **标题：** Awesome Egocentric Vision（上游仓库 README.md）
- **维护方：** sun254667（联系人 Yuanliang Sun）
- **类型：** repo / awesome-list / curated-index
- **原文：** <https://raw.githubusercontent.com/sun254667/awesome-egocentric-vision/main/README.md>（2026-11 经代理获取，1698 行）
- **一句话说明：** 第一人称视觉与具身 AI 的策展清单，按「综述 → 10 个核心主题 → 数据集/基准/仿真器 → 智能眼镜 → 相关清单 → 关键挑战」组织，许可 CC0 1.0。
- **核对范围：** 通读 README 全文结构：目录与收录标准、§1 综述（5 篇）、§2.1–§2.10 各子节的条目标题/会议年份、各节数据集表、§3 数据集/基准/仿真器、§4 智能眼镜、§5 相关清单、§6 关键挑战表、致谢与许可；§2.1–§2.2、§2.7–§2.8 的 Highlights 与数据集表逐条读过，其余小节浏览条目标题与要点。清单所链接的各篇论文/数据集本身未逐一打开，其结论仅按 README 的转述记录。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **定位与收录标准：** 学习友好的第一人称视觉与具身 AI 资源清单（论文、数据集、代码、模型、工具、挑战赛、教程）；优先 CVPR/ICCV/ECCV/NeurIPS/ICLR/AAAI/TPAMI/TVCG；覆盖 Ego-VLM/LLM、Ego-VLA、3D/4D 重建、手物交互、注视、长视频、世界模型。
- **§1 综述：** Challenges and Trends in Egocentric Vision（四类任务：主体/物体/环境/混合理解）、Egocentric Procedural AI Assistant（错误检测/流程学习/VQA 三任务）、Ego-Exo 跨视角协同综述（三范式）、IJCV 2024 展望、2012 年 First-Person Vision 奠基综述。
- **§2 十个主题：** 2.1 动作识别/检测/预期；2.2 手物交互；2.3 Ego-VLM 与多模态推理（含 EgoVLP/EgoVLPv2 预训练）；2.4 长视频与结构化推理（场景图、记忆压缩、EgoSchema）；2.5 注视与意图；2.6 3D/4D 重建（人体/手位姿、事件相机、SLAM/场景图）；2.7 Ego-VLA；2.8 Ego 世界模型；2.9 程序化助手（错误检测、流程学习、QA、流式主动交互）；2.10 数据安全与隐私。
- **§2.7 Ego-VLA（与机器人最相关）：** 条目包括 EgoScale（> 20,854 小时带动作标注的人类第一人称视频，人类数据与机器人性能之间的 log-linear 缩放）、EgoSteer（9.6K 小时灵巧 VLA 预训练全栈）、HumanScale（第一人称人类视频可优于真机遥操作数据做预训练）、EgoBridge（最优传输对齐，模仿成功率较基线 +44%）、HumanEgo（分钟级人类示范零样本）、潜在动作（Motion-Focused Latent Action）与世界动作模型（EgoWAM、OSCAR）等。
- **§2.8 Ego 世界模型：** PlayerOne、EgoSim、EgoHOI、EgoExo-WM（用第三人称视频增强）、Walk through Paintings（预训练视频扩散改造为动作条件 WM）、ENACT 基准等。
- **数据集表：** 通用锚点为 Ego4D（3,670 h / 923 人 / 9 国）、EPIC-KITCHENS-100（100 h / 90K 片段）、Ego-Exo4D（1,286 h）、HD-EPIC（41 h）；手物与灵巧：EgoDex（829 h）、HOI4D、EgoTouch（触觉）；VLA 评测：EgoVerse（1,362 h / ~80K episode）；世界模型预训练：HumanNet（~1M h）、EgoVid-5M 等；仿真器：PlayerOne、EgoSim。
- **§4 智能眼镜：** 基准（SUPERGLASSES、Project Aria 数据集）与低功耗端侧感知（OpenGlass、EgoTrigger）、终身记忆助手（EGAgent、LightMem-Ego）、可穿戴交互（VisionClaw 等）。
- **§5 相关清单：** Awesome-Egocentric、Awesome-Egocentric-and-Exocentric-Vision、Building Egocentric Procedural AI Assistant 等。
- **§6 关键挑战：** 四个通用难点——视野窄/自遮挡/抖动、数据稀缺与隐私、跨人分布偏移、超长序列建模；并按 10 个方向给出开放问题与 2026 进展（如 Ego-VLA 的人机具身差距与动作空间错位、WM 长 rollout 时序一致性）。
- **致谢与许可：** 交叉引用 Awesome-Egocentric-Video-Datasets 等社区清单；CC0 1.0。

## 对 wiki 的映射

- [`wiki/entities/awesome-egocentric-vision.md`](../../wiki/entities/awesome-egocentric-vision.md)
- 入库快照见 [awesome-egocentric-vision.md](awesome-egocentric-vision.md)
