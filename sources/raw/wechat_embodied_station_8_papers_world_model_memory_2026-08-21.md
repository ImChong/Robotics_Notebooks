---
title: 机器人不再只会“看见再行动”：8篇论文揭示世界模型与长期记忆新拐点
author: 具身智能小站
date: "2026-08-21 09:00:00"
source: "https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g"
---

# 机器人不再只会“看见再行动”：8篇论文揭示世界模型与长期记忆新拐点

---

📅 2026年8月21日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 8 篇近期机器人与具身智能论文，内容覆盖双臂抓取、灵巧操作、长期场景记忆、水下视觉、流式语音、助残喂食、全身控制与通用世界模型。看似分散的方向，其实都在回答同一个问题：当观测不完整、环境持续变化、任务链条变长时，机器人如何仍然做出稳定、可迁移、可执行的决策？

**综述主线：**能力竞争正在从单点策略精度，转向“补全隐藏状态—保存长期历史—预测动作后果—约束真实执行”的闭环系统能力。

01 · arXiv:2608.19188

🔬 **PartialBiGrasp: Inferring Hidden Local Geometry for Bimanual Graspi**ng from Partial Views

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYg8fLHh2OrAGxkpSZIhKmQSrAskLdej8jiaHSy0zNGok0rXjh06ib6pU2U4mLdevmGwXibjphHbyIyxuXIABm7q0LTiaUqia7UGqDk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 **Embodied AI · Bimanual Grasping · 3D Perception**

**✨ 一句话亮点：**不依赖完整点云，在残缺观测中推断厚度、边缘与夹爪间隙，生成物理稳定的双臂抓取对。

**📖 摘要：**大型、重型或几何复杂物体往往只有少量可抓区域，而真实 RGB-D 观测无法提供完整点云。PartialBiGrasp 直接从**局部点云**出发，借助**卷积占据网络**隐式学习**局部几何**，判断可抓性、无碰撞接触区与物体厚度；随后生成满足**力闭合**约束的抓取对，并通过**采样优化**修正不完整几何带来的歧义。论文在解析指标、大规模仿真与新物体实机实验中验证了其稳定性。

**💡 核心洞察：**双臂抓取的关键正在从“重建完整物体”转向“只补全与接触决策有关的**局部几何**”。

**🔗 项目链接：** https://partialbigrasp.github.io/

**🔗 资料来源：** https://arxiv.org/pdf/2608.19188

02 · arXiv:2608.19182

🔬 **ADEPT: Accelerating Dexterity via Pre-Training and Post-Training using Reinforcement Learning**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYuCNnyKbvtoY9Rfib5PoGuw0yXu4iaibIBNupnh8LLicicE1nSy2wSoSdf6u9plyAZiaRc7YCKIpgK4qnhMNsQE6FdecoNFAFhPN28k/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 **Dexterous Manipulation · Reinforcement Learning · Sim-to-Real · Visuo-Tactile**

**✨ 一句话亮点：**先学习**通用物体重定向**，再稳定后训练下游任务，让高自由度灵巧手从原始视觉触觉完成长视野操作。

**📖 摘要：**高自由度多指机器人很难从零探索出长视野灵巧行为，且每个新任务重复学习相同技能代价高昂。ADEPT 先在**通用物体重定向**任务上预训练策略，再把该行为作为先验后训练下游任务。为避免朴素微调破坏已有能力，方法组合**行为克隆蒸馏**、**评论家预热**与保守的 on-policy 更新，并以关节空间 **Geometric Fabric** 约束安全执行。蒸馏后的感知学生策略在两种机器人本体上实现**零样本 sim-to-real**，并以接近人类的速度完成长视野任务。

**💡 核心洞察：**灵巧操作的可扩展路线，不是为每个任务重学一遍手指技能，而是把可迁移技能先验保护好。

**🔗 项目链接：** https://adept-dexterity.github.io/

**🔗 资料来源：** https://arxiv.org/pdf/2608.19182

03 · arXiv:2608.19059

🔬 **LT-Mem: Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubXwRYR9iaRzUROmqlNysB45hnhzVFy0mpXkE5rVHXCDLJjIcpva7Rpd0MiaUTibuE7DnKwqdJho9oRWBlgsXonVuiaicB0OL9kG4Yk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 **Lifelong Scene Understanding · Spatio-Temporal Memory · SLAM · VQA**

**✨ 一句话亮点：**用**波动性**感知的三层记忆同时保留对象当前状态与历史事件，避免机器人长期运行中的“时间性失忆”。

**📖 摘要：**长期运行的机器人反复访问变化环境时，覆盖旧地图会丢失对象历史，逐次保存快照又难以维持跨会话身份。LT-Mem 将**多会话 SLAM** 对齐的对象级观测与**波动性**条件时序推理结合：确定性证据评分维持身份一致，更新策略按对象动态性选择**覆盖、保持或多假设**；**Live、Delta、Meta** 三层记忆同时记录当前状态、变化事件与元信息。配套 **LT-VQA** 数据集提供多会话记录、持久身份标注和时间问答，实验显示其全面优于基线且令牌消耗**低一个数量级**。

**💡 核心洞察：**长期记忆不是“存得更多”，而是知道什么该覆盖、什么必须保留、什么应暂存多种解释。

**🔗 项目链接：** https://lt-mem.github.io/

**🔗 资料来源：** https://arxiv.org/pdf/2608.19059

04 · arXiv:2608.18662

🔬 **Dynamic SpectraFormer for Ultra-High-Definition Underwater Image Enhancement**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZtSOwibT25T7dHkENC1odeePNFicCSkLjlJlwNtTYTVUe3NuiaFHNWv9B5vqyUic4fLCB8GklgVwnQZ6klPavOmh5GhS9U61WykL4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 **Underwater Robotics · Image Enhancement · Frequency Transformer**

**✨ 一句话亮点：**在频域同时处理**低频**色偏与**高频**纹理退化，并动态选择**关键频带**，服务超高清水下机器人视觉。

**📖 摘要：**水下光线折射与吸收会同时造成色偏、雾化和能见度下降，其中颜色与亮度失真偏**低频**，边缘与纹理失真偏**高频**，单纯空间域方法难以兼顾。Dynamic SpectraFormer 转而在频域增强图像：超高清**稀疏频谱注意**在保留通用逼近能力的同时建模长程依赖，**动态频谱权重层**则自适应强调**关键频带**、抑制次要频带。论文通过多组消融与多个水下图像增强基准验证了方法有效性。

**💡 核心洞察：**面向 AUV 的视觉增强，频域并非单纯降算力技巧，而是与水下退化机制天然对齐的表示空间。

**🔗 项目链接：** https://github.com/arifence2024/DynamicSpectraFormer.git

**🔗 资料来源：** https://arxiv.org/pdf/2608.18662

05 · arXiv:2608.18661

🔬X2Streaming-TTS: Causal Token-Level Text-to-Speech from Streaming Text with Speech-State Inheritance

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgua44LWlbJLdtEQ7BUR4gEKMRibNlKEDkBL08Z01zbGVaQrdia431KKtUicHCF3OM3sMydfAtm4VQw3T5OCxXswyXDOPmicM8JwPWts/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 **Streaming TTS · Causal Generation · Human-Robot Interaction**

**✨ 一句话亮点：**不等待完整句子，在异步令牌到达时因果生成连续语音，并用语音状态继承守住跨段自然度。

**📖 摘要：**许多所谓流式 TTS 仍需等待句级文本，真正的**令牌级合成**则要在前缀不确定、上下文受限时持续说话。X2Streaming-TTS 只消费已到达的文本令牌：**因果承诺**机制用不确定性感知缓冲与容量自适应、标点感知分段处理歧义；**因果语音状态继承**跨段携带完整 Code2Wav 状态和部分 Talker 历史状态。其在多数主客观指标上优于伪流式模型，单请求首音频令牌中位时延为 **15.8 ms**，128 并发时为 **260.8 ms**，质量接近所评估的离线基线。

**💡 核心洞察：**机器人语音交互的低时延上限，取决于系统如何管理“不确定前缀”，而不只是声学模型推理速度。

**🔗 项目链接：** https://github.com/X-Square-Robot/X2Streaming-TTS

**🔗 资料来源：** https://arxiv.org/pdf/2608.18661

06 · arXiv:2608.18258

🔬 **VERAGMIL: Virtual Environment for Scooping Granular Foods with Imitation Learning Models**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZib36ZnsmatcQgqiaRpibJ1icPyiaiclwFn3nSGNV1J8T4v0SLtYia6YGkE7ZKGjcjwvdeeGKSeaicD67ibF4NOxpyYuoRUW4pktD655aE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 **Assistive Robotics · Imitation Learning · VR Simulation**

**✨ 一句话亮点：**把高保真颗粒仿真与直观 **VR** 示范采集结合，让助残喂食机器人更稳地舀取和运输米饭、豆类。

**📖 摘要：**助残喂食机器人处理米饭、豆类等颗粒食物时，材料动态复杂，而高质量人类示范又难以获取。VERAGMIL 将**高保真仿真器**与直观 **VR** 交互界面结合，提供包含机器人、传感器和多类物理特性食物的训练环境。研究用 **VR** 与三维空间鼠标示范训练 **BC、BC-RNN、BCQ** 三类模型，并按成功率、**洒落量**、**未见食物泛化**与完成时间评估。结果显示 **VR** 示范显著优于三维空间鼠标数据，BCQ 综合表现最好，尤其能减少洒落并接近人类表现。

**💡 核心洞察：**在颗粒物操作里，示范接口本身就是学习系统的一部分；更自然的数据采集方式能直接改变策略上限。

**🔗 项目链接：** https://github.com/AmanuelErgogo/VERAGMIL.git

**🔗 资料来源：** https://arxiv.org/pdf/2608.18258

07 · arXiv:2608.18234

🔬 **GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZJiaHkPSHIq6PTUpOqGywC7v7BMMvLrcKPrd8uL7iaicFBibHqRoAHAuTBBL2dQ0X4yRhaUyvdcCBrwW4XsfMiaEo6cEA2gI1Aeaqo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 **Humanoid Robotics · Behavior World Model · Whole-Body Control**

**✨ 一句话亮点：**让控制策略同时预测动作、状态与潜在行为命令，并把不可能指令在线拉回可行行为。

**📖 摘要：**现有全身运动跟踪器多在空旷平地训练，难以理解地形和物体接触如何改变可行动作。GigaBrain-WBC-0.5 提出用于人形全身控制的**行为世界模型**：**因果 Transformer** 联合预测下一动作、下一状态与下一潜在行为命令分布；自动地形标注流程从重定向动作恢复完整**三维接触几何**；部署时再用预测分布识别不合理命令并**回撤**到已学行为。其地形交互成功率为 **81.3%**，不合理命令下为 **83.1%**，跌倒恢复为 **99.3%**，并展示了跨机器人微调迁移。

**💡 核心洞察：**世界模型开始进入低层控制：不仅预测环境，还实时判断“这条命令在当前接触条件下能不能做”。

**🔗 项目链接：** https://shepherd1226.github.io/gigabrain-wbc-0.5/

**🔗 资料来源：** https://arxiv.org/pdf/2608.18234

08 · arXiv:2608.18077

🔬 **Hydra-0: Action Flow for Generalist World Modeling and Control**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYReuhh7ouZ7XibYFgiaAz1ItBKoQ6LCuuaeGQv5dO7cQ7icMjurvQ686qDibFBPV8Et9xNSZbrOvguTxzkLqL3e5LedOiaLzVYmWSs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 **Generalist World Model · Action Flow · Multi-Embodiment Control**

**✨ 一句话亮点：**把机器人动作统一表示为**像素运动**，让人类、夹爪、双臂与单臂视频共享同一个世界建模和控制接口。

**📖 摘要：**不同机器人本体的关节与末端动作空间不统一，限制了世界模型跨数据与跨平台学习。Hydra-0 用**动作流**把机器人动作表示为图像中的**像素运动**，在本体、任务、环境与视频生成骨干之间建立**共享视觉接口**。最佳配置相较动作条件基线将机器人运动误差降低 **90.4%**、物体运动误差降低 **60.2%**，并支持零样本组合与数据高效适配；在 RoboLab 上，回放成功率与参考成功率的相关系数达到 **0.96**。其涌现出的**逆向模式**还能从人类示范的目标物体流推断兼容机器人运动，再映射为可执行动作。

**💡 核心洞察：****动作流**的价值不只是换一种控制表示，而是把异构视频、策略评估与机器人控制放进同一接口。

**🔗 项目链接：** https://nvidia-isaac.github.io/video\_to\_data/hydra-0/

**🔗 资料来源：** https://arxiv.org/pdf/2608.18077

**综合观察**

这 8 篇论文给出的信号很一致：具身智能正在把过去分散的感知、记忆、预测与控制模块重新接成闭环。PartialBiGrasp 和 Dynamic SpectraFormer先修复不完整或退化观测；LT-Mem让环境变化不再被最新状态覆盖；ADEPT、VERAGMIL与GigaBrain-WBC-0.5分别从技能先验、示范质量和行为可行性约束提升执行可靠性；Hydra-0则尝试用动作流打通跨本体世界建模与控制。真正有产业价值的系统，不只是在单次评测里“会做动作”，而是能在长期运行中持续补全、记忆、预判并自我约束。
