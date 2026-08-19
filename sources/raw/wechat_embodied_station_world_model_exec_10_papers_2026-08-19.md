---
title: 10篇机器人论文速览：世界模型很热，但真实执行才是硬门槛
author: 具身智能小站
date: "2026-08-19 09:02:00"
source: "https://mp.weixin.qq.com/s/NJ6M3CnsmDrtu9baRo8lgQ"
---

# 10篇机器人论文速览：世界模型很热，但真实执行才是硬门槛

📅 2026年8月19日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 10 篇近期**具身智能**与**机器人**论文，内容覆盖**世界模型**评测、**社会导航**、类人 **VLN**、**空间记忆**、**开词汇导航**、合成数据挑战、移动操作、**VLA** 基座模型、视频运动推理和手部**可见性**。整体来共同关注一个核心问题：**机器人**系统不只要“看懂场景”，还要在**跨本体**、**真实世界**、遮挡和长时序约束下稳定执行。

本期主线不是单点刷榜，而是从数据与评测、空间与运动表征、控制与策略学习、感知可靠性四条路径，把**具身智能**推向可诊断、可迁移、可落地的**机器人**能力。

01 · arXiv:2608.13049

🔬 ****H2R-Bench**: Benchmarking Human-to-Robot Manipulation Video Generation in World Models**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguab7Z8Gfgkr7BcL7ZtN8IdEo7FIZJib72Uc6VM1zFHxdRAcK1BFlJIMNKRkLZXrw80T9hmcU4KiaIMCzS8cybe3TFJxIyThv7cicU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 **Embodied AI · World Models · Cross-Embodiment Manipulation**

✨ 把“人类第一视角视频能否变成**机器人**训练素材”做成可诊断基准。

📖 **机器人**学习需要大规模操作数据，但真实**机器人**示范昂贵且难扩展；人类第一视角操作视频虽丰富，却存在人手与**机器人**末端执行器的**跨本体**差异。**H2R-Bench** 将人类示范视频转换为指定**机器人**本体下的操作视频，并用目标状态、动作事件、功能接触、本体正确性和视频质量五个维度评估。论文对 11 个视频生成模型、6 类操作和 2 种**机器人**本体进行评测，指出当前**世界模型**在本体一致性、功能交互和任务执行上仍明显受限。

💡 **世界模型**要真正帮**机器人**学操作，先要过**跨本体**迁移这一关。

🔗 项目链接： https://rongdingyi.github.io/H2R-Bench/

02 · arXiv:2608.12917

🔬 **Towards Socially Compliant Navigation in Deep Reinforcement Learning via Proxemics-Based Reward Modeling**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubPNBFz1iclZQk06yVAKmlds00D52L0pvAH4Vb1g9wgfoNJRf0YlSRULJAeFYmsK8q9Oibq7y3GVTcOibnsJfC7eyD0kOKGiazFIGY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 **Social Navigation · Deep **RL** · Proxemics**

✨ 用 Hall 近体学把“别贴人太近”写进 **DRL** 奖励。

📖 拥挤环境中的**机器人**导航不能只追求到达目标和避障，还要让人觉得安全、平滑、可预期。该论文提出一种基于 proxemics 的 **DRL** **社会导航**奖励，把每个人的个人空间建模为来自 Hall 近体学的径向高斯混合场，并在**机器人**视野内计算局部社会代价。作者将该奖励接入已有 **DRL** 导航方法，在多种人群场景、奖励基线和密度下评估，结果显示社会指标稳定提升，同时保持有竞争力的导航效率。

💡 **社会导航**的关键，是把人的舒适边界变成可学习的密集信号。

🔗 项目链接： https://drl-proxemics.github.io/

03 · arXiv:2608.12860

🔬 ****HumanoidVLN**: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaemuUf3ice7nlD2s1L5Kb5EeE3oEg5o7GtJMQyibmhUoHt6OCqyePcUEiaJ5evXVhq0iaAAaosfaSa16icsubbtWTuIYckgE9gcibeI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 **Humanoid Robotics · **VLN** · Sim-to-Real Benchmark**

✨ 把 **VLN** 从轮式/理想 agent 拉回双足类人**机器人**的真实约束。

📖 类人**机器人**上的 **VLN** 面临三类现有基准难覆盖的问题：双足运动带来的物理约束、不同类人本体之间的形态差异，以及行走引起的第一视角相机动态扰动。**HumanoidVLN** 基于 NVIDIA Isaac Sim 构建物理约束仿真与 benchmark，覆盖 Unitree G1、Unitree H1 等 4 种本体，并用强化学习步态策略结合 PD 或 MPC 路径跟踪器。数据包含 933 个带避碰参考的 episode，支持多种 **VLN** 模型；摘要中还报告了 Janus**VLN** 的最好均值表现，以及 Dual**VLN** 在 Unitree G1 上的 sim-real pilot 相关性。

💡 **类人导航**不是把轮式 **VLN** 换个外壳，而是模型、控制器和本体共同作用。

🔗 项目链接： https://humanoid-vln.github.io/

04 · arXiv:2608.12743

🔬 ****Spatial Memory Agent**: Experience-Grounded Procedure Memory for Spatial Intelligence**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguaXXXp68cFIk6b8Vd623vmRvh0SIc4qGvfBegddjWfSGFUSYMibmcgNPlXSKsYjNTMxMEHC6StYzrtc8XRt6jN81tBpDvWvmt4U/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 **Spatial Intelligence · Memory Agent · Frozen **VLM****

✨ 冻结 **VLM** 不调参，也能从验证过的空间经验里长记性。

📖 **空间智能**正在成为具身 agent、**机器人**规划和多模态助手的基础能力。现有方法多依赖后训练，或在推理时调用深度估计、3D 重建等外部空间工具。**SMA** 走的是互补路线：在不更新参数、推理时不依赖外部专家工具的前提下，把已验证的空间经验转化为可复用的过程性 lesson。它通过 verifier-guided reflection 写入经验，并用 Transfer Reliability Score 校准记忆可靠性；部署时按语义过滤和 similarity-TRS 排名检索记忆。摘要报告其在 5 个空间 benchmark、4 个基础 **VLM** 上获得每个 base-model block 的最高 macro average。

💡 **空间记忆**的价值，不在复读旧题，而在沉淀可迁移的操作性经验。

🔗 项目链接： https://aim-uofa.github.io/SMA/

05 · arXiv:2608.12707

🔬 ****SAP-Nav**: Spatial Semantic Representation Meets Active Perception for Hierarchical Open-Vocabulary Object Navigation**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaR1yApn1eeb9Z9Jhhe2ia6SA2RY5oPWCzGWRNfOTP97T95rQAj7qWRnAM6GWGgpqXiaxmDkDMVnKbLNptj1icZ2y5DkUZ4sW1o7s/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 **Open-Vocabulary Navigation · Active Perception · Spatial Semantics**

✨ 不靠离线地图，边走边主动找更有信息量的视角。

📖 层级化 **OVON** 要求**机器人**理解自由语言中包含的场景、房间、区域和实例级线索，而部分观测下的空间 grounding 与目标验证仍然困难。**SAP-Nav** 是一个完全在线、zero-shot 的**主动感知**框架：它从主动获取的房间视角中增量构建 Queryable Spatial-Semantic Representation，让 agent 可以从已探索位置发起空间语义查询；同时通过 Active Viewpoint Verification 判断当前视角是否足够，并在必要时移动到更具判别力的位置再验证目标。摘要称其在 LangMap 和 HM3D-**OVON** 上取得整体最好表现，region-level SR 相比训练式方法提升 12.2%，并通过真实**机器人**实验验证可行性。

💡 **开词汇导航**的瓶颈，往往不是识别词表，而是主动补足空间证据。

🔗 项目链接： https://github.com/XuetongPei/SAP-Nav

06 · arXiv:2608.12416

🔬 ****RoboSynChallenge**: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgub5XSevw0O0Fzo0ZicrQUy5XuOggRcxs0tfMZqxic7cy5sExulzYw4uUKibECWSaQhM7WibG7xBY5JBS7G3lh7DBmBVwURlNIsTTDc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 **Dexterous Manipulation · Synthetic Data · Real-World Evaluation**

✨ 用合成操作技能训练，但最终只看**真实世界**泛化。

📖 通用**机器人**操作仍受限于真实数据稀缺和多样性不足。**RoboSynChallenge** 提供一个统一 benchmark，用于评估操作策略在任务、环境和难度跨度上的泛化能力。挑战赛鼓励参赛者使用大规模合成 state-action trials 提升通用策略学习，但最终评估只在未见过的**真实世界**操作环境中进行。为保证可复现和可比较，基线覆盖 Transformer、Diffusion、**VLA** 与 World-Action-Model 等策略类型。该工作试图把可扩展仿真数据生成与严格**真实世界**验证接起来，推动更数据高效、可适应的操作系统。

💡 合成数据能不能算数，最后要由真实**机器人**泛化来裁决。

🔗 项目链接： https://github.com/EDEM-AI/RoboSynChallenge/

07 · arXiv:2608.12063

🔬 **Learning Loco-Manipulation From **SMPC** Demonstrations With Sparse Offline-to-Online **RL****

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYaebCHX0ib0zsMx4UibTNJB5UsE3PaQIMa3fSCdCTRnmibme1b3frxYnksWkjrg6UQlibhlKLqE2Wk9kWmU8NnenQGfwacBGS0DGs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 **Loco-Manipulation · **SMPC** · Offline-to-Online **RL****

✨ 让最优控制当仿真老师，用**稀疏奖励**学会复杂移动操作。

📖 移动操作把 locomotion 与 manipulation 绑定在一起，但标准 **RL** 在复杂任务上常被密集奖励手工设计拖慢。该论文用完全仿真的 Sample-based Model Predictive Control 作为快速可调的 expert，自动生成大规模离线数据，从而先解决探索问题；随后用纯稀疏任务奖励训练 off-policy **RL** agent，减少新技能学习所需的人工调参。高层 agent 与低层动态稳定控制器结合后，学到的行为可以更贴近真实任务目标，甚至超越原始最优控制老师。摘要还报告了该 sim-to-real 框架在带机械臂的 Spot 四足和 G1 类人**机器人**上的部署。

💡 移动操作里，**SMPC** 更像可扩展老师，而稀疏 **RL** 负责把目标学透。

🔗 项目链接： https://pages.rai-inst.com/smpc2rl/

08 · arXiv:2608.11739

🔬 ****Galaxea G0.5**: One Autoregressive Stream for Robot Reasoning and Action**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZutImhOmHAaUNkAFwgFQz8pSDXADy8oOwIGZUgRTKdsKLiapWjX1fK2cM42gjuSicdH4gff3LEiaRXXkVHz6cSRMCOicgnic8U8Pzk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 **Vision-Language-Action · Autoregressive Policy · Robot Foundation Model**

✨ 把推理和动作放进同一条自回归 token 流。

📖 当前常见 **VLA** 配方通常把预训练 **VLM** 与独立的 flow-matching 动作专家耦合，**VLM** 更像上下文编码器而不是决策者。**Galaxea G0.5** 反过来聚焦 **VLM** backbone：单个 transformer decoder 在同一目标下生成推理 token 与动作 token。摘要提出三项关键组件：**跨本体**动作 tokenizer、交织任务分解/目标 grounding/动作 hint 的 chain-of-thought 流，以及通过视觉编码器注入多秒历史的 visual memory。由于推理和动作共享权重，模型可通过 prompt 直接影响动作粒度、任务跨度和 OOD 场景处理；摘要还列出其在真实**机器人**微调、BEHAVIOR Challenge、DROID zero-shot transfer、LIBERO、RoboTwin 2.0 等 7 类 regime 上超过已有方法。

💡 **VLA** 的一个重要方向，是让语言推理和连续动作共享同一套决策接口。

🔗 项目链接： https://opengalaxea.github.io/G05/

09 · arXiv:2608.11655

🔬 ****Motion-as-Prompt**: Enhancing Motion Reasoning in Multimodal Large Language Models via Motion-Guided Cross-Frame Visual Prompting**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZzr4uu5fEchCIp8Ld2qCu6jVsuRichJDia0UshJQOIf1OquQwHjSnzJ0XJ0wcF3ovPHNHPJQzicAbLetQ7ZplAFdyk3Sp1UUAapo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

📌 **Video Reasoning · Motion Prompting · **MLLM****

✨ 不改模型，把帧间运动轨迹直接画给 **MLLM** 看。

📖 **机器人**操作和自主导航都依赖运动中心的视频推理，但 **MLLM** 为控制 token 与注意力成本，常用稀疏均匀采样处理视频，导致关键帧间转移、碰撞和因果互动被丢掉。**Motion-as-Prompt** 是一个 track-guided cross-frame visual prompting 框架：它恢复密集点轨迹，选择运动信息量高的帧，并把相邻采样帧之间累积的轨迹直接标注到视觉输入上，让冻结 **MLLM** 看到原本隐藏的位移、方向变化和交互。摘要报告其在 CLEVRER 和 Something-Something-v2 上提升平均运动推理准确率，GPT-5.5 分别获得 4.2% 和 8.9% 增益，且不损害非运动理解。

💡 有时提升视频推理，不必改模型，先把运动证据显式化。

🔗 项目链接： https://github.com/SunVictor23/MaP

10 · arXiv:2608.11574

🔬 ****Hand Visibility Detector**: Per-Keypoint Visibility Estimation for Hands**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguauF909ZsJ04eiaiaBJKF3G6Cc2OCNe9eLqqzIfrzAnr263cnfWYX8WFtjRTZjbdxlicQI1ibLKicxT5TO2aPv490fQEiakdOpkCgNGA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

📌 **Hand Pose Estimation · Visibility · 3D Annotation**

✨ 手部姿态不只要点位，还要知道每个关节到底看不看得见。

📖 手部姿态估计是 AR/VR、人机交互和**机器人**中的基础技术，但多数 HPE 方法输出关键点位置时，并不显式说明每个关节在图像中是否可见。该论文提出 **Hand Visibility Detector**，将 per-joint hand visibility estimation 作为独立任务系统研究，而不是仅作为姿态估计的辅助信号。方法利用大规模手部数据预训练 HPE 模型的先验作为 backbone，并展示其在**可见性**估计上表现良好；同时，在多视角 2D 关键点三角化生成 3D 手部姿态标注的下游任务中，visibility-weighted triangulation 可降低 reprojection error。

💡 **机器人**理解手部操作，可靠性信号和坐标本身同样重要。

🔗 项目链接： https://github.com/ryhara/hand\_visibility\_detector

**综合观察**

信号很清晰：**世界模型**正在被要求跨过**人到机器人**的本体差异，导航研究开始把社会距离、类人动力学和**主动感知**纳入闭环，**VLA** 与**空间记忆**在探索不改模型或统一 token 流的推理-行动路径，控制侧则用 **SMPC** 示范和合成数据缓解**稀疏奖励**与真实数据不足。下一阶段更值得关注的，不是单个模型是否会说会看，而是它在真实**机器人**上的空间证据、动作接口和可靠性估计是否足够闭环。

**资料来源**

01. https://arxiv.org/pdf/2608.13049

02. https://arxiv.org/pdf/2608.12917

03. https://arxiv.org/pdf/2608.12860

04. https://arxiv.org/pdf/2608.12743

05. https://arxiv.org/pdf/2608.12707

06. https://arxiv.org/pdf/2608.12416

07. https://arxiv.org/pdf/2608.12063

08. https://arxiv.org/pdf/2608.11739

09. https://arxiv.org/pdf/2608.11655

10. https://arxiv.org/pdf/2608.11574
