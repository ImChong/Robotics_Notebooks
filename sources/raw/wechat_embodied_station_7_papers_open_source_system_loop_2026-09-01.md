---
title: "机器人圈开源速递：LightNav-0、Zeva、Motus2等7篇新作"
author: 具身智能小站
date: "2026-09-01 14:01:00"
source: "https://mp.weixin.qq.com/s/IkK6lFCu4hjBX0sA1hMqgA"
---

# 机器人圈开源速递：LightNav-0、Zeva、Motus2等7篇新作

📅 2026年9月1日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 7 篇 最新**机器人**与**具身智能**论文，覆盖语言驱动 **Quality-Diversity**、通用具身导航、在线因果学习、灵巧操作**世界模型**、**VLA** **推理期纠错**、**多智能体****机器人**策略，以及复杂光学场景**单目几何**。整体来看，这一批论文的共同关键词不是单点刷榜，而是把模型、记忆、验证、感知和**项目入口**一起推向**可复现**。

**综述主线：****具身智能**正在进入“**开源**入口 + **系统闭环**”的阶段：从自然语言生成技能档案，到 **VLM** 导航、**因果记忆**、**世界模型**自演化，再到无需训练的语言纠错和复杂场景几何感知，**机器人**能力开始更强调可调用、可验证、可迁移。

01 · arXiv:2608.30983

🔬 **Autonomously Acquiring Robot Manipulation Skills with Language-Driven **Quality-Diversity****

📌 **Embodied AI · Robot Learning · **Quality-Diversity** · **LLM****

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguahduKD28F3d8HpK7Guc2ZEPxKf7rGpNhL23uaR6rlPy7GXVgK7Bjyribcd9hiaFSTnBDTupnfiaXZjfKJaia6e9PgE2hibBCz94CuE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 只需**自然语言**任务描述，自动生成多样化机器人运动技能档案。

📖 **Quality-Diversity** 方法能为机器人构建多样**运动原语库**，使其在部署时**零样本**适应约束，但通常依赖专家手写成功条件、适应度和多样性度量；**LLM** 奖励塑形虽能提升自主学习，却多产出单一高性能解。本文提出语言驱动 **QD** 框架，仅需自由形式任务描述，即可自动探索适应度与**行为描述符**函数空间，并结合 multi-BD **MAP-Elites** success 算法生成多样运动原语档案。基于 **Genesis** 仿真的 4 个操作任务实验显示，该方法优于使用推断或手写参数化的经典 **QD** 算法。

💡 把任务语言转成技能档案，比只生成一个最优策略更适合真实部署。

🔗 项目链接： https://github.com/EGarrabe/Language-driven-robotic-QD/tree/main

🔗 资料来源： https://arxiv.org/pdf/2608.30983

02 · arXiv:2608.30935

🔬 ****LightNav-0**: Eliciting **VLM** Spatial Intelligence for Generalist Embodied Navigation**

📌 **Embodied AI · Embodied Navigation · **VLM** · Generalist Robot**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgua7evsIb48ABXmLzDVrETZ76mgud9L6bzf5CEumLkwEJ2FdnBukvVSFibR8rnKIDslfxNnY5TpMuoanCUox5RvmTibciaYdh8BIEM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 用**统一 token** 接口激发 **VLM** 空间智能，覆盖多任务**具身导航**。

📖 **具身导航**要求智能体在不同任务、环境和机器人本体中，把异构目标与视觉观测转成动作；但现有系统常依赖任务或本体专用组件，泛化受限。**LightNav-0** 是一个紧凑的通用**具身导航**模型，通过**统一 token** 接口对齐预训练 **VLM** 的空间智能：**dual-channel pointing** 表达任务、场景和本体无关的空间意图，residual vector-quantized **action tokenizer** 将其映射为精确轨迹。结合视觉历史压缩、ER mid-training、SFT 与 RL，该模型支持指令跟随、开放词汇目标导航和视觉跟踪。训练语料覆盖 **2K+** 场景与 **4K+** 小时数据，**LightNav-0** 在 10 个公开导航仿真设置中达到 **SOTA** 单目成功率，并在真实场景中展示跨本体**零样本**泛化。

💡 导航模型的关键，不是再接一堆头，而是把空间意图统一编码。

🔗 项目链接： https://www.lightorigins.com/en/blog/lightnav-0https://github.com/lightorigins/LightNav-0https://huggingface.co/LightOriginsHQ/LightNav-0

🔗 资料来源： https://arxiv.org/pdf/2608.30935

03 · arXiv:2608.30880

🔬 ****Zeva**: **In-Context** Causal Learning for Generalizable Embodied Manipulation**

📌 **Embodied AI · Causal Learning · In-context Learning · **VLA**/**WAM****

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgua2j6cPCib3vFWbQnGTiaqPA6tzvOm2QpUXria4ib9laMpXjOTWICtKYwnufnMo4OIJ5SmDHgibY6NwziaLkmsvWdHFzsqqvEsvXW3Y4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ **冻结策略**模型，让机器人在部署中从自身交互经验继续进化。

📖 仅靠预训练很难覆盖真实世界中的未知物理条件，机器人需要在部署时从自己的物理交互中即时学习，并用这些经验指导后续动作。**Zeva** 是一个在冻结 policy model 前提下实现机器人自有交互经验 in-context learning 的框架。它通过 **Causal Interaction Extractor** 将已执行动作及其导致的状态变化编码为因果交互信号，并存入 **dual-timescale causal memory**；后续动作生成时，相关信号会从记忆中检索并注入**冻结策略**模型上下文。仿真和真实操作实验显示，**Zeva** 在对比的前沿 **VLA** 与 **WAM** 中表现最好，并能在无梯度更新的部署过程中随经验积累持续提升成功率，交互经验还可**跨任务泛化**。

💡 真正的在线适应，可以先从上下文记忆而不是权重更新开始。

🔗 项目链接： https://air-embodied-brain.github.io/Zeva

🔗 资料来源： https://arxiv.org/pdf/2608.30880

04 · arXiv:2608.30237

🔬 ****Motus2**: A Self-Evolving General World Model for Dexterous Manipulation**

📌 **Embodied AI · World Model · Dexterous Manipulation · Tactile Sensing**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZqKlDAs4TqRjoxdSDmagyso9jHqicW7M9t9KKCvshbOHfspHeT8QwVXXoB2sojmTulbhJz4vE5EjYkG3JNiax2NVW6MIC24CmV8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 把 **policy**、**simulator**、**evaluator** 融成一个**自演化****灵巧操作****世界模型**。

📖 通用具身智能体需要在统一系统中感知、预测、行动、评估并改进，而现有**世界模型**常只是给世界模拟器附加动作输出头，缺少闭环策略提升机制。**Motus2** 提出面向**灵巧操作**的**自演化**通用**世界模型**：单个共享权重模型暴露 **policy**、action-conditioned world model 和 **value model** 三个控制接口，**policy** 生成候选动作片段，**simulator** 预测视觉后果，**evaluator** 评估结果，从而形成决策与学习闭环。数据扩展路径从大规模单目第一视角数据，到同步双目第一视角数据，再到机器人轨迹和人机对齐数据的机器人域适配；系统还研究全局自回归、混合记忆上下文、**触觉反馈**，并落到**双臂****双灵巧手**仿生平台。

💡 **世界模型**的下一步，是让预测、评估和策略改进真正闭环。

🔗 项目链接： https://motus-robotics.github.io/motus2

🔗 资料来源： https://arxiv.org/pdf/2608.30237

05 · arXiv:2608.29967

🔬 ****Training-Free** Action Correction for **VLA** Model Failures via Language Feedback**

📌 **Embodied AI · **VLA** · Language Feedback · Training-free Correction**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguaBltOEatBsohWvoBx5KDYRKsz6BUNzx6LO6ib2PgiaBh4yJeDMIz4ic23WQ8Ficbbur25xRVmCyTmDA6dPicG6cbOjjTSJbziamjq20/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 不用重训权重，把任务级**语言反馈**转成**动作幅度**修正。

📖 **VLA** 模型具备较强语义理解能力，但部署时仍会出现系统性失败；这些失败在什么条件下发生、能否不重训修正，仍缺少清晰边界。**Correct**VLA**** 将任务级自然语言纠正转化为加性**动作幅度**调整，不修改策略权重；人类只需提供一次任务级反馈，该修正会在所有 rollout 中统一应用。仿真中，**Correct**VLA**** 能恢复分布内与 **OOD** 任务中的执行错位失败；真实 UFactory **xArm7** 环境迁移实验中，它在基础策略几乎失效时恢复接近完美的成功率。基于 **LIBERO-90** 的失败模式分类显示，目标正确但**动作幅度**失准的 **execution misalignment** 是可修正子集，而语义理解本身崩溃的失败并不适用。

💡 **推理期纠错**有效，但前提是策略已经“想对了”。

🔗 项目链接： https://correctvla.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.29967

06 · arXiv:2608.29896

🔬 ****EMERGE-Policy**: A **Robot Mind** Emerges Beyond a Single Policy**

📌 **Embodied AI · Agentic Robotics · **Multi-agent** Orchestration · Skill Interface**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZvKeELot2BvS7eZUicmKlW3ibATy8EKiaia5rAQp6DsGqlpRbgGjVJfH1zwGNEAoOrbLjOkrW6Ggnk8iawU4WNTeeGUjr5mvGOvTWs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 把感知、执行监控、验证和记忆拆成子智能体，形成系统级机器人策略。

📖 论文提出，一个机器人的有效“心智”未必存在于单一 policy 中，而可以从多个专门组件在共享编排流程中的协作涌现。**EMERGE-Policy** 是一个图结构 agentic framework，用于协调能力调用与信息交换：**Main Agent** 在活动上下文中保持任务级状态，感知、执行监控、验证、记忆整合等 **Sub Agents** 在隔离上下文中处理专门任务并返回结构化证据。框架通过角色上下文控制信息负载，以 Operational、Imagination 和 **Evaluation Skills** 组合异构后端，并利用基于准则的验证、文本失败诊断、**Branch Stack** recovery 与 token-aware **external memory** 做局部纠错和状态保持。无需额外微调，该系统在多个公开 benchmark 和真实机器人实验中展示了突出的系统级表现。

💡 机器人策略可以是编排出来的，不必全部压进一个网络。

🔗 项目链接： https://emerge-policy.github.io/EMERGE-Policy/https://github.com/EMERGE-Policy/EMERGE-Policy

🔗 资料来源： https://arxiv.org/pdf/2608.29896

07 · arXiv:2608.29881

🔬 ****OptiGeo**: Efficient Monocular Geometry for Embodied Perception in Optically Challenging Scenes**

📌 **Embodied AI · Embodied Perception · Monocular Geometry · **Depth Estimation****

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYgpMVmmBnvdhnic8XNTKKmsntEDwCdjrq19u9Mqr5VaZMec0Z4TLCyjCP4w1ic2KrQQich62lv5Bt6YApJ8QXqbCRbfgp1OcuYYI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 用偏差感知训练修复**透明**、**反光**、**镜面**场景中的**单目几何**失真。

📖 单目深度估计已具备较强开放域泛化能力，但在**透明**、**反光**和**镜面**环境中，深度传感器常产生缺失或偏差，机器人部署仍不可靠。现有方法常使用场景特定预处理、辅助模块或事后微调，容易带来结构冗余和窄场景过拟合。**OptiGeo** 将问题重新定义为基础模型训练中的局部失败模式，指出 **sensor-induced supervision bias** 是关键瓶颈，并提出 bias-aware training：用 **clean-geometry teacher** 和 **residual-trimmed alignment** 修复有偏真实监督，同时把**透明**目标渲染视作紧凑的干净光学几何来源。仅凭小规模 targeted rendering set，**30M** 参数的 **OptiGeo** 在**透明**场景 benchmark 上超过更大的 **300M** 单目模型和十亿级多视角 baseline，并在**真实导航**案例中验证其实用性。

💡 真实机器人的几何感知，必须正面处理传感器偏差。

🔗 项目链接： https://mx-liu6.github.io/OptiGeo-web/

🔗 资料来源： https://arxiv.org/pdf/2608.29881

**综合观察**

综合来看，这 7 篇论文给出的信号相当一致：**机器人**基础模型不再只拼一个大 policy，而是在探索更**可复现**的系统结构。Language-driven QD 让自然语言任务描述生成多样技能档案；LightNav-0 把 **VLM** 的空间智能直接接入导航；Zeva 和 Motus2 分别从**因果记忆**与**世界模型**闭环推进自适应；Correct**VLA** 给出推理期语言纠错的边界；EMERGE-Policy 把**多智能体**编排变成系统级策略；OptiGeo 则提醒我们，真实部署里透明、反光、镜面这类感知长尾依然重要。对圈内人来说，真正值得收藏的是这些**开源**/**项目入口**背后的**可复现**实验路径。
