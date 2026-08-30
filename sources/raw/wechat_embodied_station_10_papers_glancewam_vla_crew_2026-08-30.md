---
title: 48ms世界模型来了！10个开源项目串起VLA、多机协作与人形机器人
author: 具身智能小站
date: "2026-08-30 09:00:00"
source: "https://mp.weixin.qq.com/s/MdCtmijSM_VfYp19f-nZQw"
---

# 48ms世界模型来了！10个开源项目串起VLA、多机协作与人形机器人

📅 2026年8月30日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 10 篇近期机器人与具身智能论文，覆盖异步世界动作模型、行为意图蒸馏、多机器人智能体编排、双臂 VLA、模仿能力评测、四足移动操作、人形机器人世界模型、约束规划、仿真到现实认证与模块化人形系统。它们共同回答一个现实问题：当模型走出离线数据集，怎样兼顾泛化能力、实时控制与可验证的物理可靠性？

**综述主线：**本期工作的共同趋势，是把关键结构从隐式学习变成显式接口：GlanceWAM 将想象移出控制关键路径，INDI 让动作解码器恢复行为目标，Physical Agentic AI 与 Meta-Ctrl 把执行约束写进系统架构，Bet4Sim2Real 用模拟器组合收紧真实性能区间；M3、DreamMimic、TONAV 和 GOLEM 则分别从模态扰动、预测表征、导航—操作衔接和模块化集成补强真实部署。

01 · arXiv:2608.23927

🔬 **GlanceWAM: Sparse Test-Time Imagination for World-Action Models**

📌 **World-Action Model · Asynchronous Inference · Video DiT · Robot Manipulation**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubgow5hjuMwbofic8c0YVkLncnm3UMnEAGkKK8OJvVChibnPCKUyAbYiatbNYaDHZm8NdtcbQBPjYWtYvY0PsksWg8L2OQTsuucGc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 把视觉想象移出控制关键路径，以**异步稀疏前瞻**同时保住实时性和任务成功率。

📖 世界动作模型若按控制频率同步生成视频，延迟难以接受；完全取消测试时视觉想象，又会损失任务成功率。GlanceWAM 在单个视频 DiT 内解耦想象与控制：异步 proposer 以较慢频率在后台生成数秒后的单帧前瞻，动作头则直接在潜空间以 **48ms** 解码动作块，不被视频生成阻塞。非干扰注意力掩码与抗陈旧时域训练进一步适应前瞻老化。仅用示范训练后，模型在 RoboCasa 24 项任务和 LIBERO 上分别达到 **72.2% 与 99.0%**，在 A100 上比同步基线快 24 倍。

💡 世界模型的部署瓶颈未必是生成本身，而是**生成是否阻塞控制**。

🔗 项目链接： https://github.com/linhanwang/GlanceWAM

🔗 资料来源： https://arxiv.org/pdf/2608.23927

02 · arXiv:2608.23478

🔬 **Act with Intent: Distilling Behavior Intent for Vision-Language-Action Models**

📌 **Vision-Language-Action · Intention Distillation · Behavior Cloning · Multimodal Learning**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaLOLoazribK4djP1xEZFMcudpU6HiaiaU9vQ4sjqtTNntF9b03ia6ohjZ1g3C0hLUQ5hpdy19nRjEOoBAYhR15FzGlnXxHqqDdQds/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 不只模仿电机命令，而是把**行为级语义意图**蒸馏进 VLA 动作解码器。

📖 行为克隆告诉 VLA 应复现哪条电机指令，却没有显式监督该行为在当前指令下服务的局部目标；未来帧或轨迹监督也更偏向某次具体实现。INDI 在训练时让冻结教师 VLM 根据当前观察、指令、粗粒度动作摘要和执行视频解释行为意图，再让部署端 VLA 在动作解码器中间层恢复这一多模态意图表征。方法将 GR00T-N1.7 在 SimplerEnv-Bridge 上从 **64.3% 提升至 84.7%**，在 RoboCasa Kitchen 上从 64.1% 提升至 70.3%；真机平均成功率由 62.0% 提升至 **68.7%**。

💡 动作预测若显式理解“为什么这样做”，更容易组织**目标一致的后续行为**。

🔗 项目链接： https://leesangoh.github.io/indi-project-page/

🔗 资料来源： https://arxiv.org/pdf/2608.23478

03 · arXiv:2608.22657

🔬 **Physical Agentic AI: An Architecture for Orchestrating a Robot Crew with LLMs**

📌 **Agentic AI · Multi-Robot Systems · LLM Planning · Safety Verification**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguaJ7T6SicUbUyG9teWfX08kTbH00dSugHD2Of7k9zE7sn2QS6bHdbrOAxMlprm0vnk7hlb0uP5QFYX7oicq39VwDUln7Ymj3n3K8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ 将语义规划与物理执行分离，用**确定性编排器**逐项验证多机器人技能调用。

📖 即便向 LLM 提供本体能力、物理前提和跨机器人协作信息，规划仍可能包含不可行、错时或不安全动作。Physical Agentic AI 让机器人暴露带类型的可执行技能库，由无执行权限的 Mission Planner 分解任务并分配机器人—技能对，再由 Robot Orchestrator 根据机器人状态、位置和工作流约束验证后逐项执行。作者在无人机—UGV 仿真任务及人形—四足真机搬运中发现，检索虽改善技能落地，错误派发率仍高于 20%；确定性编排器则将错误派发降至 **0%**，并阻止全部注入故障。

💡 LLM 可以负责提出计划，但物理系统的**最终执行权必须可验证**。

🔗 项目链接： https://github.com/Liuuuxy/physical-agentic-ai

🔗 资料来源： https://arxiv.org/pdf/2608.22657

04 · arXiv:2608.22419

🔬 **Robust Bimanual Vision-Language-Action Models via Embarrassingly Simple Modality Masking**

📌 **Vision-Language-Action · Bimanual Manipulation · Modality Masking · Robustness**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZqczOKN4nD20R2rZc5veiaBngygsqLeKxsm5lXeSia1mTibP4z12kVarLPrXH23E9IObvR7VdsQd4XJzJWqICKPxDEsfVicVLulib4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 仅在训练期随机屏蔽模态通道，无需改结构即可增强**双臂 VLA 的多视角鲁棒性**。

📖 查询式 VLA 具有低延迟优势，但复杂双臂任务中仍会出现动作不连续和执行失败，原因之一是多视角与语言融合不稳定、注意力被干扰区域分散。M3 是一种无需架构修改或大规模机器人预训练的训练期策略：随机遮蔽部分模态通道，以受控的不完整观察迫使策略减少对干扰线索的依赖。在 RoboTwin 2.0 的 10 个双臂任务上，相较 Adapter 基线，M3 在 Clean 和 Clean2Rand 设置下平均成功率分别提高 **21.7% 和 11.4%**；3 个长时程真机任务的平均完整任务成功率也提升超过 30%。

💡 鲁棒多模态融合有时不靠增加模块，而靠训练时**主动制造信息缺失**。

🔗 项目链接： https://m3vla.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.22419

05 · arXiv:2608.22301

🔬 **The Imitator Game: Benchmarking Robot Imitative Ability Beyond Action Prediction**

📌 **Robot Imitation · Benchmark · Human Video · Intent Understanding**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguY8a7Oib9bckx6FIiciaMXNkxsd9lIiaIPeQZwRKdJqFfyKaaR7ic7mdF8goLLH2m3JVbZWkjIab42Jz040575MJuwicdKLBVxHO07EY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 用四级任务差异测试机器人是在复现轨迹，还是理解并迁移**示范者的真实意图**。

📖 现有机器人策略多学习视觉和语言到动作的映射，面对人类视频时往往只能在近似场景中复现轨迹，难以用不同工具或物体实现同一目标。The Imitator Game 构建 L0–L3 四级基准，逐步拉大人类示范与机器人现场的差异；配套 IG-10K 包含 **2 万余组人机配对、50 余项任务和 6 个领域**，并提供开放盲测平台 Imitator Arena。9 个先进模型在 L0–L2 表现稳定，却在要求功能替代的 L3 明显崩溃；所有模型在未见任务上的零样本成功率均低于 13%，但仅用 10 组配对示范微调即可获得显著增益。

💡 机器人真正的模仿能力，应以**目标等价而非动作相似**来衡量。

🔗 项目链接： https://imitator-game.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.22301

06 · arXiv:2608.22296

🔬 **TONAV: Task-Oriented Navigation and Action-Velocity Chunk Learning for Articulated Object Quadrupedal Mobile Manipulation**

📌 **Mobile Manipulation · Quadruped Robot · Vision-Language Reasoning · Action Chunking**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguaNI8qg5X4JBUZichrAHSY717RmoNSmgUkib8FqT8G52IW2fEK1KlZDsaicerib3KqJg3JcjBs7GcxO1icBliaDZaI55UiaYllKO1gthM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 把任务导向导航与**位置—速度动作块**统一起来，填补抵达目标与稳定操作之间的空档。

📖 四足移动操作既要到达适合操作的构型，又要在铰接物体交互中保持稳定接触；现有方法常在“靠近目标”后结束导航，导致可达却不可操作。TONAV 以位置—速度耦合遥操作采集平滑示范，再用视觉语言推理把高层指令拆成可执行子目标，并持续调整机器人底座至操作就绪位姿；动作—速度块学习则联合建模关节位置及其时间变化，以速度监督改善持续接触。多类真实铰接物体实验表明，该方法提升任务导向导航和完整移动操作成功率，并缓解跟踪滞后、动作抖动与接触不稳。

💡 移动操作的关键不是“导航后再操作”，而是让导航从一开始就**服务于接触任务**。

🔗 项目链接： https://haochen611.github.io/TONAV

🔗 资料来源： https://arxiv.org/pdf/2608.22296

07 · arXiv:2608.22278

🔬 **DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation via World Model**

📌 **Humanoid Robot · World Model · Teacher-Student Distillation · Loco-Manipulation**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZGUbIFIbBRQRE2hLK99I2Z4XCdVGjmdgsMIlISzYNNsGt9SMXaMPTu7OPlprJmeE5tZW3fhsiaaD5TBFZYmAEweBz2xibd1OvZM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 把世界模型用作**预测表征与多步监督器**，稳定人形机器人的视觉全身运动操作。

📖 视觉全身运动操作同时面临部分可观测、丰富接触和长时程误差累积。DreamMimic 通过世界模型辅助蒸馏，把特权教师策略迁移到视觉人形控制器：RSSM 不用于在线规划，而是学习预测潜在动力学，为学生提供表征空间、动作条件多步监督和紧凑预测特征；特权状态、接触、物体状态与奖励辅助头强化交互信息。Performance-Conditioned Guidance 根据师生表现动态平衡指导和探索。在 OMOMO 与 BEHAVE 上，该方法优于强视觉基线，部署时学生不接触在线特权状态。**GitHub 当前标注 Codes coming soon。**

💡 世界模型不仅能规划未来，也能成为视觉策略蒸馏的**稳定监督空间**。

🔗 项目链接： 项目页GitHub（代码即将开放）

🔗 资料来源： https://arxiv.org/pdf/2608.22278

08 · arXiv:2608.22149

🔬 **Meta-Ctrl: Guaranteed Plan Generation by Decoupling Syntactic and Semantic Constraints**

📌 **Robot Planning · Constrained Decoding · Large Language Model · Formal Guarantees**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYcSibYtMia4ic4RqZF4gXpiaLR3uEqSqeKkX8lV4faQjFmSdlec6IExicswEcHVkJ5JyhiaIxCvBiczfZhERnnmAMsyGJwuq8QJJHSEY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

✨ 用**元令牌与双层约束解码**保证机器人计划合法，同时保留语言模型的常识与质量。

📖 LLM 能生成流畅的机器人计划，却经常违反执行所需的语法和语义约束；软约束没有保证，符号规划又容易丢失模型常识。Meta-Ctrl 引入由落地动作组成的紧凑 meta-token 词表，在令牌层保证语法，在动作层执行前置条件、目标和顺序约束。精确因式分解把受约束解码的内存需求从超过 **107TB 降至 2GB 以下**。借助该框架，小型开放权重语言模型在 LoTa-Bench 的 WAH-NL 上取得最高已报告子目标成功率并超过 GPT-4；真实桌面机器人生成的每个计划也都按构造满足前置条件与目标。

💡 可靠规划无需在常识与保证之间二选一，关键是**分离两类约束**。

🔗 项目链接： https://meta-ctrlg.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.22149

09 · arXiv:2608.21572

🔬 **Betting for Sim-to-Real Performance Certificates**

📌 **Sim-to-Real · Performance Certificate · Statistical Guarantee · Robot Evaluation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubz5ItJ7Kw9wFfOjvtQ1icdB6HTqoHQyWIjGRiaJ109dALFE79v2yuvPYuZqoCib24IiaTbFiaqbeKnicHia4O00FJia3NsiccofMuJ2xEU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

✨ 把模拟器结果变成逐次下注，用**随时有效的统计证书**减少昂贵真机测试需求。

📖 机器人真实性能通常用少量真机试验估计均值及置信区间，但样本昂贵会使证书过宽。该工作提出 sim-to-real betting certificate：每次真实结果揭晓前，算法参考大规模模拟结果组合下注，真实结果结算财富并动态调整对不同模拟器的信任，再把累计财富转化为性能区间。理论上，返回证书对任意模拟器库都保持 anytime-valid，并以财富遗憾界指导配置。合成分布和真实机器人测试显示，相比经典及先进基线，证书平均收窄 **51.6%±16%**；在不超过 30 个样本时仍收窄 **32.26%±8%**。

💡 仿真价值不只在训练策略，也能用于**量化地减少现实评测不确定性**。

🔗 项目链接： https://github.com/ISUSAIL/Bet4Sim2Real-Certificate

🔗 资料来源： https://arxiv.org/pdf/2608.21572

10 · arXiv:2608.21550

🔬 **GOLEM: Modular Humanoid Autonomy Towards Electric Vehicle Battery Disassembly**

📌 **Humanoid Robot · Open-Source System · ROS 2 · Industrial Disassembly**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZ7c8lTicUMhWF8XfbufRdzD5q2UHbBaB3f17WEbveNJ3xa3e3MnbZLOAX2zmL9GIcACzWPZTxxlwdqKEesvmtib3s8ouENBbd7c/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

✨ 以**模块化开源架构**打通人形机器人的行走、操作、导航和动力电池拆解。

📖 退役电动车电池拆解单调且危险，目前主要依赖人工。GOLEM 是面向 Unitree H1-2 的端到端开源系统，将行走、操作、动态稳定、导航和空间记忆拆成具有抽象接口的独立模块，并以 Docker 化 ROS 2 连接具有一致接口的 MuJoCo、IsaacLab 数字孪生和真机。系统采用能力阶梯逐项评估模块：LiDAR—惯性导航在 6 米目标上达到 **13.0cm** 定位误差；学习式站立控制器能恢复采样式下肢 MPC 无法应对的扰动；真实 Ioniq 5 电池包紧固件抓取成功率从系留 97% 降至自由站立 87%，加入导航位姿扰动后为 37%。

💡 人形机器人进入工业现场，需要先建立**可替换、可比较的系统模块**。

🔗 项目链接： 项目页与源代码

🔗 资料来源： https://arxiv.org/pdf/2608.21550

**综合观察**

综合来看，具身智能的下一阶段并不是单纯把视觉语言模型做得更大，而是重新设计决策链条：想象何时发生、行为目标如何进入解码器、多个机器人由谁验证动作、模拟结果如何转化为现实证书、模块怎样在仿真与真机之间保持一致。开放资源也呈现多层次形态：部分论文已提供代码或系统，部分开放数据集、评测平台与项目页，DreamMimic 当前则明确为代码即将发布。复现时应区分论文结果、项目演示和实际可下载资产。
