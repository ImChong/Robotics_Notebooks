---
title: 具身智能开源资源集中上新：9篇论文，WAM、VLA、跨本体一次看全
author: 具身智能小站
date: "2026-08-28 09:00:00"
source: "https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ"
---

# 具身智能开源资源集中上新：9篇论文，WAM、VLA、跨本体一次看全

📅 2026年8月28日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 9 篇近期开源具身智能论文，覆盖人类视频驱动的世界动作模型、流式 VLA、跨本体统一策略、语言推理、多臂协作、三维世界建模、主动学习、鲁棒里程计与建筑机器人任务分类。整体来看，这些工作共同关注一个核心问题：如何让机器人从单任务、单形态的策略，走向能够理解意图、复用知识并适应真实环境的通用系统。

**综述主线：**这一批工作的共同主线，是为机器人策略加入更明确的结构化接口：Zero-WAM 用人类视频定义新任务，StreamPI 引入持续时间记忆，UCAG-P 以相机坐标中的动作几何对齐不同本体，R^3 与 MA-VLA 分别显式组织语言推理和多臂子目标；GaussianDream++、ConfAL-WM 与 SUPER ODOMETRY 2.0 则从三维预测、置信度和传感器退化三条路径补强部署可靠性。

01 · arXiv:2608.26103

🔬 **Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization**

📌 **Embodied AI · World-Action Model · In-Context Learning · Human Video**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgua7X4iaGAbO4DJyZcsUvbYTF0dU19JCNlrexs27ia2sCjjFBfhPzVaCmwogZKiac7WlwKbUo4RNI7YGI3FfcDbnPcA71ibEpbovXkI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 把**人类视频直接变成任务说明**，让机器人无需参数更新即可执行训练中未见的新任务。

📖 零样本跨任务泛化要求策略处理训练阶段从未出现的操作任务，而语言往往不足以完整描述物体交互过程。Zero-WAM 将大模型的上下文学习范式引入机器人操作，以人类视频提供任务演化线索，并通过自动流程构建包含 **7.42 万个人机配对、覆盖 8600 个任务**的 HumanGen 数据集；其 IFP 训练目标迫使模型从视频提示中提取任务信息。在 RoboTwin 2.0 的 7 个未见任务上，平均成功率达到 **47.0%**，比最强视频动作基线高 **29.5 个百分点**，真实机器人也展示了多物体、长时程和精细插入任务泛化。

💡 当**视频成为任务规格**，机器人泛化可以从重新训练转向上下文理解。

🔗 项目链接： https://robbyant-research.github.io/Zero-WAM/

🔗 资料来源： https://arxiv.org/pdf/2608.26103v1

02 · arXiv:2608.26067

🔬 **StreamPI: Streaming Multimodal Temporal Modeling for Vision-Language-Action Models**

📌 **Vision-Language-Action · Streaming Inference · Temporal Modeling · Multimodal**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYMCcR67bU54SwzfsSFCPXiaMgKpqZiaPAEJxia4myER4PJJCsqsicXbf1jUtMZFYNgjicxNdtlhvic9LdTV5yUfPVRk8PjYylgia9GrA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 无需增加参数，为单帧 VLA 注入**可流式推理的时间记忆**与异步部署能力。

📖 以 π0.5 为代表的单帧 VLA 难以保留历史观察，也不利于精细空间感知。StreamPI 将每个“视觉观察—语言指令”对视作原子时间单元：单元内部用双向注意力融合模态，单元之间采用因果注意力维持流式推理，让指令持续充当语义锚点。随机间隔流式训练进一步弥合同步训练与异步真机执行的差异，并提升对帧间隔扰动的鲁棒性。该方法不新增参数，可继承单帧预训练权重并灵活进行单帧或多帧推理，在记忆依赖、精细感知真机任务及 LIBERO 上均优于 π0.5。

💡 VLA 的时间能力未必需要更大模型，关键在于**注意力结构与部署节奏一致**。

🔗 项目链接： https://happinesslz.github.io/projects/StreamPI

🔗 资料来源： https://arxiv.org/pdf/2608.26067v1

03 · arXiv:2608.26058

🔬 **One Policy, Many Embodiments: Unified Camera-Centric Action Geometry Pre-training for Heterogeneous Embodied Manipulation**

📌 **Vision-Language-Action · Cross-Embodiment · Action Geometry · Pre-training**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYzQ3PGJ96YPehibTTgHyjeNXS0GEmAQiaqY2UpaicgvMZ2RWbvvbSNCyVH5YZCWhhulNeNrfY0EVERaD0AJuCEUq6Q1Klt2FNXkY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ 以**相机可观测的锚点运动**统一机器人手臂、人形机器人和人手的异构动作空间。

📖 不同机器人形态、相机配置和底层动作空间限制统一 VLA 的联合训练。UCAG-P 不把本体专属控制量作为共享目标，而是在相机坐标系中用可观测锚点运动表达操作，再由几何条件动作转换器结合目标本体运动学生成控制。该解耦设计让共享策略学习可迁移的操作几何，同时保留本体特定控制能力。模型使用 **4030 小时机器人与仿真数据、2340 小时人类示范**训练；单一检查点在 LIBERO、RoboTwin Easy/Hard、LIBERO-Plus 零样本和 RoboCasa GR-1 上分别达到 98.3%、88.7%/89.2%、82.0% 和 62.0%。

💡 跨本体学习的突破口，可能不是统一关节，而是统一**可观察的动作几何**。

🔗 项目链接： 项目页与代码

🔗 资料来源： https://arxiv.org/pdf/2608.26058v1

04 · arXiv:2608.26053

🔬 **R^3: Training Robots to Reason in Natural Language via Reinforcement Learning**

📌 **Robot Reasoning · Reinforcement Learning · VLM · Long-Horizon Manipulation**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZJcoAlmdg4qZ3Rp05zDS7AiaNCZnqRd00U2sXSlHafS7BYgOf7iaeY7r1l1kRfUG9TzhN9myMHicdJnugscpy1DsFqU43fbmzOQo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 让 VLM 在行动前进行**自由形式自然语言推理**，用测试时计算引导低层操作策略。

📖 长时程操作需要追踪局部进度、物体关系和动作后果，并在低层策略出错后恢复，但自然语言推理能否真正改善机器人操作仍缺乏直接验证。R^3 提出一套简洁的后训练流程：先用专家生成的推理轨迹进行中期训练，初始化目标推理风格，再利用离线动作数据执行单步、基于量表奖励的强化学习。不同于把结构化推理轨迹仅当辅助监督的方法，R^3 直接训练自由形式语言推理，并把它作为动作策略的测试时指导。在 Language Table 和模拟双臂杂货打包任务上，该方法提升了探索与未见任务泛化，并显著优于仅指令模仿学习基线。

💡 语言推理的真正价值，是成为低层策略可调用的**测试时计算接口**。

🔗 项目链接： https://robotic-reasoner.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.26053v1

05 · arXiv:2608.25864

🔬 **MA-VLA: Multi-Arm Vision-Language-Action Model for Collaboration and Compositional Generalization**

📌 **Vision-Language-Action · Multi-Arm Collaboration · Compositional Generalization · Open Source**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaiaExrlAmg1axicfEXzicY2gHYdyHfeOXe6Mw6xYOUmJGSgXuQN419Zia4KC9aNmwTAl3mqurz8E6YXYU8PiazKdZA1g5q4ymjianmw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 通过**逐臂原子动作分配**与 Arm Shuffle，让多臂协作摆脱固定执行角色。

📖 现有 VLA 通常把语言表示为一条全局指令，缺少向不同机械臂分配并组合专属行为的机制，因此难以迁移到训练中未出现的协作模式。MA-VLA 将协作行为拆成中层原子提示并分配给各机械臂，实现显式子目标定义与跨任务组合复用；训练阶段的 Arm Shuffle 同步置换每条机械臂的观察、状态和原子提示，迫使策略学习与角色无关的指令跟随。作者还构建了测试协作模式不出现在训练集中的基准；仿真与真机结果显示，既有先进 VLA 在此设置下大多失败，而 MA-VLA 能持续完成任务。论文明确开放**代码、模型与数据**。

💡 多臂通用化的核心，不只是共享感知，而是让**角色与技能可重新组合**。

🔗 项目链接： https://github.com/zhangzaibin/future-robots

🔗 资料来源： https://arxiv.org/pdf/2608.25864v1

06 · arXiv:2608.25659

🔬 **GaussianDream++: Efficient 3D Gaussian World Modeling for Robotic Manipulation**

📌 **World Model · 3D Gaussian Splatting · Vision-Language-Action · Robot Manipulation**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguae1iarqjCWsBzhTMHDiatBQ8Zx7kJnFafhoYzyibkFznNUgNEYibtLzBG6gS3LtEvU7ia1UvicxwnTc4M6rp0N5XSibh4WmuuGGia5gtI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 把当前世界与未来预测压缩进**20 个世界令牌**，推理时无需在线高斯解码。

📖 动作模仿对度量三维结构和短期物理演化监督有限，预测式策略又常增加部署成本。GaussianDream++ 在 VLA 主干中插入 World State Tokens 与 World Prediction Tokens，训练期将其解码为共享高斯基元上的当前世界和未来预测，以静态—动态分解聚焦交互区域。推理时移除表征头、渲染器、辅助目标及 VGGT/TGE 路径，仅保留 20 个世界令牌。方法在 LIBERO 和 LIBERO-Plus 上达到 **98.6% 与 87.8%**；真机平均成功率由复现 π0.5 的 29.2% 提升至 **52.5%**。

💡 三维世界监督可以只在训练期存在，把部署收益留在**策略内部表征**中。

🔗 项目链接： GitHub项目页

🔗 资料来源： https://arxiv.org/pdf/2608.25659v1

07 · arXiv:2608.25572

🔬 **ConfAL-WM: Confidence-Guided Active Learning for Action-Conditioned World Models**

📌 **World Model · Active Learning · Confidence Estimation · Synthetic Data**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYoy2OzWK7jib8xgAGExfVWmSicE9BKEkDGxCbnPZjDnh3PqHHGTubka6pjmb8rPicpJZCkg79E6q4icvwZjqmk8xQicb5g9asyNiaIo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 用**稠密置信度风险图**同时决定世界模型该学哪些数据、重点修正哪些区域。

📖 动作条件世界模型在新任务和新场景中的误差，往往集中于机械臂、操作物体、接触区和遮挡目标等局部时空区域。ConfAL-WM 面向世界模型后训练，在 EVAC 的 UNet 解码特征上增加轻量置信度探针，预测潜空间稠密置信度图，并聚合为任务、帧和图块三级评分。流程先用少量目标域数据重训探针并预热模型，再进行任务级预筛选分配采样预算，最后用已选数据结合可选的帧或图块加权增强训练。RoboTwin 2.0 实验显示，该选择策略提高后训练效率，稠密加权也比标量奖励、进度及评审式评分基线带来更好的预测质量与具身轨迹一致性。

💡 世界模型的主动学习应从“选样本”升级为**定位风险并分配监督**。

🔗 项目链接： https://ConfAL-WM.github.io

🔗 资料来源： https://arxiv.org/pdf/2608.25572v1

08 · arXiv:2608.25427

🔬 **SUPER ODOMETRY 2.0: Resilient Odometry via Hierarchical Adaptation**

📌 **Robot Odometry · Sensor Fusion · Hierarchical Adaptation · Resilient Autonomy**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubWFYQI9nlc4d85RbkuWRtvXib3VeQAnMdC9Evjan0GTOhIfiaNCywsFT9iaGm0tVRSSCaO4xtGIibo3VKeD24ZkE7ElcpFSAwrgcg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

✨ 通过四级自适应传感器融合，在烟雾、沙尘、积雪与弱光中保持**韧性定位**。

📖 复杂动态环境中的烟雾、沙尘暴、积雪和弱光会严重削弱外感知传感器，使现有里程计漂移甚至失效。SUPER ODOMETRY 2.0 提出分层自适应传感器融合框架，由自适应特征选择、状态方向选择、引擎选择和学习式惯性里程计四个模块逐级提升适应能力。其惯性模型使用超过 **100 小时**的异构机器人数据训练，并将 IMU 提升到与相机和 LiDAR 同等重要的位置，在外感知失效时作为可靠后备。系统已在空中、轮式和腿式机器人上完成 **200 公里、800 小时**验证，覆盖多种传感器配置、退化环境和激进运动。

💡 可靠定位不是寻找永不失效的传感器，而是建立**可退化、可切换的层级体系**。

🔗 项目链接： https://superodometry.com

🔗 资料来源： https://arxiv.org/pdf/2608.25427v1

09 · arXiv:2608.25395

🔬 **A Taxonomy of Construction Task Activities for Robot Workers**

📌 **Construction Robotics · Task Taxonomy · Skill Library · Vision-Language-Action**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubHzI8IwsHZgGiaicFMrgyicBkKibBn7hZiaI74pS02RVqE2qhejPqUQJfCYvnmjzMgy5KCPeyGxbKZFFvOicMWyicBnOZvu4NArM9ANU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

✨ 用**41 个动作原语**建立建筑作业共同词表，把人类工种活动连接到机器人技能库。

📖 VLA 为通用机器人提供了扩展任务范围的可能，但建筑现场首先需要准确盘点工人活动及其能力要求。TARCAT 是一套职业任务驱动的分类体系，来源包括 7 个高就业建筑工种的 **91 项 O\*NET 任务**和 30 段实体作业教学视频。它将 41 个动作原语组织为 12 个组和 3 个类别，并支持把带参数的原语序列组合成可复用技能。这一可解释结构可用于整理示范、定义机器人能力需求，以及支持编码智能体检索和扩展技能库。作者还在搭载 CRAFT 手的 DOBOT CR3 机械臂上展示了部分原语，并开放标注。

💡 建筑机器人走向通用化之前，行业首先需要一套**人机共享的任务语言**。

🔗 项目链接： https://github.com/AICPS/TARCAT-Taxonomy

🔗 资料来源： https://arxiv.org/pdf/2608.25395v1

**综合观察**

这 9 篇论文的价值不只在于刷新单项指标，更在于把具身智能的关键接口逐步显式化：视频可以成为任务说明，时间上下文可以持续流入策略，动作可以在本体之外用统一几何表示，语言推理可以分配测试时计算，置信度可以直接指导世界模型补课。与此同时，开放资源的形态也在分化：部分项目已提供代码、模型或数据，部分目前以项目页和视频为主。阅读和复现时，建议将“论文结果”“项目演示”和“可下载资产”分开判断。
