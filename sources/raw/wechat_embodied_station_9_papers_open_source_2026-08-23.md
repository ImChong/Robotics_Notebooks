---
title: 9篇开源论文看懂具身智能新动向
author: 具身智能小站
date: "2026-08-23 09:00:00"
source: "https://mp.weixin.qq.com/s/CXOf3PU8-H6OzI77vnhZMA"
---

# 9篇开源论文看懂具身智能新动向

📅 2026年8月23日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 9 篇 具身智能论文，内容覆盖长视野 **VLA**、动作分块、具身基础模型、流匹配策略、人机介入强化学习、灵巧手力觉重定向、多摄像头推理、运动规划加速与在线 **RL** 探索。整体来看，这批论文都在回答同一个问题：机器人策略如何从离线数据、长上下文和真实交互中获得更强的可执行性与鲁棒性。

**综述主线：**从「更长的上下文」到「更可控的动作分布」，再到「真实硬件反馈」，具身智能正在把策略学习从静态模仿推向可诊断、可适配、可闭环的系统。

**速览地图**

长视野 **VLA** 与动作分块具身基础模型规模化真实交互与力觉反馈规划、监测与闭环探索

01 · arXiv:2608.16172

🔬 ****SparkVLA**: Stop-Aware Hierarchical **VLA** with Adaptive **Action Chunking** for Long-Horizon Manipulation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubib0Gk7AkR3Qw5F9GdywQ82iaNWwk9VnJjuqMbwYW809kDTVfJdxd3quPtlRFSEmbic5b163svm4OT7b5Yfqht9jUvAZRBUR7vE8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 **Embodied AI · **VLA** · Long-Horizon Manipulation · **Action Chunking****

✨ 把停机判断和动作 chunk 长度统一排序，提升长视野操作决策

📖 层级 **VLA** 在每个重新观察点都要决定何时结束当前子任务、以及执行多长 action chunk，但两者相互依赖，现有方法常分开判断。**SparkVLA** 将 Stop 与所有 action-prefix length 放入统一候选集排序，减少阈值调参；同时用 Anchor-Conditioned Context Encoding 缓存子任务锚点，并由 Stop-Aware Action-Prefix Selection head 高效评分。RoboCerebra 上成功率达到 47.12%，超过官方层级 baseline 30.57%，真实机器人多步任务也验证了收益。

💡 长视野 **VLA** 的接口决策，不能再把停止和执行长度拆开看。

🔗 项目链接： https://icr-lab.github.io/SparkVLA

🔗 资料来源： https://arxiv.org/pdf/2608.16172

02 · arXiv:2608.15938

🔬 **Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYMhnBC6YoCJmKpC2p9qAOmWwagnw1FDmk0rtQUYGOibePAEcoQWzxaQ0FgKYvOIs1LkibUwGg0AuAhE3vcoyrYzJw2qxxFb1oW8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 **Robot Imitation Learning · **Action Chunking** · Reactivity · Closed-Loop Policy**

✨ 重新解释 **open-loop** action chunking，为长上下文闭环策略正名

📖 动作分块通过预测动作序列并 **open-loop** 执行前缀，已成为机器人模仿学习的重要技巧，但长 **open-loop** 前缀会削弱反应能力。该论文认为，长 **open-loop** 执行主要帮助短上下文策略模仿 non-Markovian demonstrations，而不是简单缓解 compounding errors。四个仿真和两个真实任务显示，专家演示的非马尔可夫性强烈影响成功率与执行 horizon 的关系；当策略获得足够长上下文后，**open-loop** 不再有益，最具反应性的 **closed-loop** 策略表现最好。

💡 动作分块的红利，可能来自补短上下文，而不是放弃闭环。

🔗 项目链接： https://revisiting-open-loop-action-chunking.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.15938

03 · arXiv:2608.15875

🔬 ****GigaBrain-0.7**: Scaling Embodied Foundation Models to Emergent Capabilities with a **Three-System Architecture****

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYNZwpm5uqY9UAL575rbY3TAjjlpgmna8WxXnvLKW8Hwia4HH9Loer8v9t4voSCcTUcuTu2cje85MicLTm3OiaopV3pia9oZZpiav3g/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 **Embodied Foundation Model · **VLA** · World Model · Multi-Embodiment**

✨ 用三系统架构和 3.7 万小时异构数据扩展具身基础模型

📖 **VLA** 已成为通用具身智能体的主流范式，但能否通过更有效架构、更大异构数据和跨任务跨本体泛化继续扩展，仍是开放问题。**GigaBrain-0.7** 以 **Three-System Architecture** 统一理解、预测与动作，预训练规模超过 37,000 小时异构具身数据，并采用 one-stage alignment 同时优化视觉语言理解和多本体动作生成。相较 **GigaBrain**-0 系列和包括 π0.5 在内的已有模型，它在 zero-shot 基础能力、语言条件指令跟随和 post-training 任务成功率上显著提升，并在家庭与工业场景展示适应能力。

💡 具身基础模型的扩展，正在从单策略走向系统架构工程。

🔗 项目链接： https://gigaai.cc/blog/gigabrain07 ｜ GitHub: https://github.com/open-gigaai/giga-brain-0

🔗 资料来源： https://arxiv.org/pdf/2608.15875

04 · arXiv:2608.15748

🔬 **Making two action heads agree: coordination mechanisms and a **runtime collapse certificate** for **flow-matching** policies**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZvhd5hdfO0XFiaEhSgQm9obibjwc1YVpahib3uDqJicQrR6a9ZEbRBgOA1RcIZZDG5vpOCJLCdgw5bbibJR6KSo6GTEJCpWmOKrbZg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 **Flow-Matching Policy · Robot Monitoring · Multimodal Action · Runtime Certificate**

✨ 研究双动作头如何协调，并给出区分协调与坍缩的运行时证书

📖 双表示 **flow-matching** policy 会把同一预测运动解码到关节空间和末端空间，二者残差可作为物理可解释的运行时信号；但在多模态任务中，两个分支可能各自选择不同有效模式，导致误报。论文系统比较多种协调机制：共享辅助 latent 会在总体最优中被擦除；共享 source noise 可协调也可反协调，取决于表示映射；一致性正则获得中等协调但降低有效配对率；训练支持的离散 partition token 则稳健接近上限。作者还提出无需真值标签的 collapse certificate，在 zero mismatch 含义不明时区分协调与坍缩。

💡 机器人策略监测要区分真异常和多模态下的合理分歧。

🔗 项目链接： https://github.com/kimo423/dual-head-coordination

🔗 资料来源： https://arxiv.org/pdf/2608.15748

05 · arXiv:2608.15741

🔬 **Some Modifications to Our End-to-End UAV Planner**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubeJTOUeK6cXdWWvA2LUPwFd1qPHdpX3LxibjbEOHRVPWEHcfjEyF1dENtaZlpz7r31dibR9evF5lpWTvFTdXtnllLHuYfWP8QPY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 **UAV Planning · End-to-End Planner · **MINCO** · Homotopy**

✨ 改造 **YOPO**：两段 **MINCO**、多同伦预测和排序损失提升飞行规划

📖 **YOPO** 这类 one-stage planner 从深度图和机器人状态直接输出候选轨迹，并通过可微轨迹代价反传训练，但会继承软约束优化的典型问题：安全代价与平滑、到达目标相互竞争，跨 **homotopy** class 非凸，单段多项式表达能力有限。该报告总结多项有效修改：采用 two-piece **MINCO** 参数化，在不改变空间轨迹轮廓下以时间换平滑；将多模态预测扩展到不同 **homotopy** classes；加入速度、加速度 barrier penalty 与曲率相关限速；并用 ranking loss 替代分数回归。结果带来更丰富轨迹表示、更安全避障和更直接路径。

💡 端到端规划仍需要把几何结构和优化约束显式放回系统。

🔗 项目链接： https://github.com/TJU-Aerial-Robotics/YOPO/tree/YOPO-MINCO

🔗 资料来源： https://arxiv.org/pdf/2608.15741

06 · arXiv:2608.15707

🔬 ****GAINS**: Leveraging Inconsistent Human Intervention Signals in Reinforcement Learning**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYUmSezQKv9ia49RknoYLGfc1IO1skL9ks9OpssWiaAVKphicjUbBtjibbF8191LKs6pG3daSVETvMafibnFpBIjqPGuDv5ib7LvjLWc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 **Reinforcement Learning · Human Intervention · Robot Manipulation · Safety**

✨ 把人类干预信号的不一致性建模进 **RL**，提升安全和恢复能力

📖 通过人类干预纠正机器人操作策略，有利于真实部署，但操作者在动作和干预时机上都并不完美；高控制频率下，干预信号常有延迟，并在时间与状态空间中不一致。**GAINS** 使用 **distributional RL** 和 quantile Q-networks 建模稀疏奖励与不一致干预引起的 return variability，并基于该分布表示设计 pessimistic exploration，以在人类纠正下安全且样本高效地学习。四个仿真操作任务和两个真实场景中，**GAINS** 比 **RL**IF 任务成功率高 22%，失败场景恢复成功率最高提升 43%。

💡 人类反馈不是完美标签，而是需要被建模的噪声信号。

🔗 项目链接： https://gains-hil.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.15707

07 · arXiv:2608.15560

🔬 ****ReForce**: Learning Force-aware Retargeting for Dexterous Manipulation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZRZWW5537ZicHyJeld0TEA0g6PJQ2icmpN26F5hozFXibnd9PGpCfAKibvzeic7L5r15ic7bthCS59NG1icNAQ2wSBKJrr2yZ0l60Sxc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 **Dexterous Manipulation · Force Feedback · Retargeting · Teleoperation**

✨ 从运动重定向走向力觉重定向，让人类示范更贴近真实接触

📖 人类示范是灵巧操作的重要数据来源，但由于 embodiment gap，将其转成机器人动作并不容易；现有 retargeting 多偏运动学，而操作成败往往由接触力决定。**REFORCE** 是 **force-aware** retargeting 方法，将人类运动和力转成能复现目标接触的机器人动作。它在运动学重定向动作上预测 residual，以达到期望力，并使用大规模仿真交互训练的通用 force tracker 支持在线 **force-aware** teleoperation 和离线数据翻译。仿真与真实硬件上，**REFORCE** 在纸杯抓取、夹钳操作等接触丰富任务中降低力跟踪误差，并增强多指接触参与。

💡 灵巧操作的数据迁移，核心不只是像人动，还要像人接触。

🔗 项目链接： https://wuyuhang-eai.github.io/reforce/

🔗 资料来源： https://arxiv.org/pdf/2608.15560

08 · arXiv:2608.15440

🔬 **Accelerating Mixed Discrete-Continuous Motion Planning via Neural **Graphs of Convex Sets****

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZD5jdY05iaz2vibag3tU1YMnOUB3piav4QDljUW5CYiczfianicW7ia2hhWtkAAh7KgJHxnfYdFLGo8bxCQwic8SCs5x3ePu3mA0j9hfg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 **Motion Planning · **Graphs of Convex Sets** · Graph Neural Network · Contact-Rich Manipulation**

✨ 用 GAT 替代昂贵凸松弛，让 GCS 在线重规划快两个数量级

📖 无碰撞导航和接触丰富操作可自然表述为耦合离散决策与连续轨迹的优化问题，**Graphs of Convex Sets** 通过图节点表示离散决策、边表示连续轨迹，是一种实用框架，但其优化子问题对在线重规划来说开销较大。该论文提出 learning-based 加速策略：用 **Graph Attention Network** 单次前向替代 nominal GCS 中昂贵的 convex relaxation，预测高概率候选路径，再由轻量 ranking network 按估计轨迹代价排序，依序评估并提前终止。方法在 3D 四旋翼、7-DoF 机械臂和 planar pushing 等任务上验证，在保持 100% 成功率的同时相较 nominal GCS 最高带来两个数量级加速，但解存在一定次优性。

💡 学习模块可以先替规划器筛路，而不是直接取代规划器。

🔗 项目链接： https://neural-gcs.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.15440

09 · arXiv:2608.15139

🔬 ****StructRL**: Structured Action-Space Exploration for Flow-Based **VLA**s**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguakUwUUviaWSdJDgCPh5pldJ0icEql1wpNHjiczbKIaR4VlTh5Dt1Mbb66NCVIG5BMzVDa5MjbgsISyBF13QMY9ckC4R1DAaOW7XA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

📌 ****VLA** · Online **RL** · Flow Policy · Structured Exploration**

✨ 把结构化随机性移到动作空间，缓解 flow-based **VLA** 探索稀释

📖 Flow-based **VLA** 已广泛用于连续机器人操作，在线 **RL** 正成为适配新任务的重要手段。现有 **RL** 方法常在 denoising chain 内注入随机性，但机器人探索需要时间平滑、并按动作组缩放的结构化噪声；直接把链内噪声换成结构化形式也不够，因为中间 flow time 注入的噪声会被后续去噪步骤削弱，论文称之为 **Structured Noise Dilution**。**StructRL** 通过确定性 ODE decoder、直接在动作空间注入结构化噪声、以及 last-step replay，将策略随机性绑定到最终执行动作，并提供可训练信号。三个 flow-based **VLA** 模型、多仿真基准和两个真实任务显示，它相较链内 baseline 提升探索效率和 OOD 表现。

💡 **VLA** 在线适配的探索噪声，必须作用在真正执行的动作上。

🔗 项目链接： https://flyfaerss.github.io/structrl/

🔗 资料来源： https://arxiv.org/pdf/2608.15139

**综合观察**

这 9 篇论文的共同脉络，是把机器人学习的薄弱环节往系统层推进。**SparkVLA**、**open-loop** action chunking 与 **StructRL** 都在重新审视动作 chunk、闭环响应和探索噪声；**GigaBrain-0.7**、**CrossView** 则把多模态与多视角理解放到更大尺度；**GAINS**、**ReForce**、**Neural GCS** 和 **YOPO**-**MINCO** 说明真实机器人仍离不开人类介入、力觉反馈、规划结构和安全约束。圈内真正值得追踪的，不是单一模型名，而是这些机制能否在多机器人、多任务、多场景中稳定复用。
