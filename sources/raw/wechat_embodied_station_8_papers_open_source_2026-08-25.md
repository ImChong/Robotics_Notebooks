---
title: "机器人圈开源清单：8篇新作，太空采矿、Q-Planning、视觉触觉全齐了"
author: 具身智能小站
date: "2026-08-25 09:00:00"
source: "https://mp.weixin.qq.com/s/71jZDzvcWZ3SsoHOEA8sgQ"
---

# 机器人圈开源清单：8篇新作，太空采矿、Q-Planning、视觉触觉全齐了

📅 2026年8月25日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 8 篇近期机器人与具身智能论文，覆盖太空采矿、视觉—触觉物理属性感知、策略自改进、异构群体导航、人机教学、物理探索、触觉安全与第一视角手部运动恢复。它们共同追问的是：机器人如何从“看见并模仿”，走向理解物理世界、主动获取信息，并在真实部署中持续变得更可靠。

**综述主线：**具身智能的能力边界正从单一策略扩展为一条完整闭环：以多模态感知理解物理属性，以价值函数和主动探索改进行动，以结构化控制保证安全，再用开放代码、数据与研究清单加速复现和验证。

01 · arXiv:2608.21358

🔬 **Mining beyond Earth with Space Robots: Exploration, Sampling, and Extraction**

📌 **Space Robotics · Autonomous Mining · ISRU · Survey**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguahg7HJGicDUD8a4lbKemx0ic3I33rB8pq03fdiaTxowicyQXlHuq3MQd7vZR51IeDrOKiaCaL51wW9gW3OCb6INFQ91tegibIzb5ASc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 提出覆盖**勘探—采样—提取**的六阶段太空采矿架构，并开放持续更新的研究资源库。

📖 太空资源利用是长期载人探索与太空商业化的重要基础，但极端环境、通信时延和高发射成本使**高自主机器人系统**成为关键。本文系统梳理太空采矿背景、国际政策、商业实体与技术进展，并将全流程划分为六个阶段：遥感选址、原位精细探测、单机器人小规模采样、多机器人规模化挖掘、自主资源提取，以及原位建造或地面运输。作者还整理真实任务数据、地球类比数据集与高保真仿真环境，并给出面向自主太空采矿的挑战与研究路线图。

💡 太空采矿真正的门槛不是单台机器人，而是**跨阶段自主系统与验证基础设施**。

🔗 项目链接： https://github.com/OpenSpace-Lab/Space-Mining-with-Robotics-List

🔗 资料来源： https://arxiv.org/pdf/2608.21358

02 · arXiv:2608.21355

🔬 **ViTacPhys: Physical Property-Aware Grasping from Human Visual-Tactile Demonstrations**

📌 **Visual-Tactile Learning · Physical Properties · Adaptive Grasping · Human-to-Robot Transfer**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZjZApB29emzwCNXaDHorW65icZ0vHDEoy4w2UAXZTbz0uKm8wz8ODiaV6EHyIwD1dRk5QN1boa98tv5WcYS3drUib6C3awykdYqw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 从人类视触觉示范显式预测**质量、刚度与摩擦类别**，再据此自适应抓取。

📖 现有视觉动作模型擅长复杂操作，却很少显式利用物体物理属性调整策略。ViTacPhys 构建视触觉框架与采集系统，从 60 个刚性和可变形物体的人类示范中预测质量、刚度与摩擦系数类别，并结合时序视触觉建模、跨注意力多模态融合和 VLM 语义先验。在已见物体上，质量与摩擦分类准确率分别为 **97.2%** 和 **98.8%**，刚度 MAPE 为 **5.51%**；迁移到机器人后，物理属性条件抓取策略在 ID 与 OOD 物体上的总成功率分别达到 **95.0%** 和 **83.4%**。

💡 把物理属性变成策略的显式条件，是从“识物”迈向**因物施力**的关键。

🔗 项目链接： https://vitacphys.github.io/ViTacPhys/

🔗 资料来源： https://arxiv.org/pdf/2608.21355

03 · arXiv:2608.21204

🔬 **Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning**

📌 **Robot Learning · Off-Policy Q-Learning · Self-Improvement · VLA**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYyO1jsNYsDdokGTd6r6iczH2heV5X5lI7XeVgbRVTJDVyRNLxMduEa4kRKvgtib1zDGKspMYfYMeXTNFWKWmso4GM4l4iakDiao7w/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ 冻结大规模模仿策略，只训练小型 **Q 函数**，让机器人同时从成功与失败中自我改进。

📖 行为克隆依赖成功示范，却无法自然地从部署失败中继续学习；直接用强化学习微调大规模机器人策略又难以扩展。Q-Planning 为大型视觉运动行为克隆策略配备小型离策略 Q 函数：推理时对候选动作块进行价值引导，在线阶段仅用成功与失败轨迹更新 Q 函数，保持原策略冻结。十轮自改进将 **LIBERO-10 从 93% 提升至 99%**、RoboTwin 从 83.8% 提升至 91.4%；在真实双臂任务中，叠杯由 40% 升至 90%，插钱包由 25% 升至 80%。

💡 自改进未必需要重训巨型策略，**价值层的轻量更新**可能更具工程可扩展性。

🔗 项目链接： https://varungiridhar.github.io/qplanning/

🔗 资料来源： https://arxiv.org/pdf/2608.21204

04 · arXiv:2608.21175

🔬 **SRL-MPC: Shape-Aware Reinforcement Learned Model Predictive Control**

📌 **Robot Navigation · Shape-Aware Control · Reinforcement Learning · MPC**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYiahAFgIABYVnd1Vhg5n6uoTuXny4I2SvzDpBZt7GF5SMEGceIlQRL2dcahRPUClxYnK6icdFia9vqYGxzsoVHFuu1ZybriaZYlQw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 用强化学习在线调节 MPC，同时以**显式形状安全约束**应对异构密集机器人群。

📖 异构人群与机器人集群中的安全高效导航，常受同质形状、稀疏场景、几何简化和手工调参等假设限制。SRL-MPC 不简化几何形状，而是基于支撑函数变换得到几何分离特征，并构造高阶控制屏障函数约束；强化学习策略读取这些特征，实时更新 MPC 参数。该设计保留 MPC 的安全结构与泛化能力，同时引入学习方法的适应性。随机生成的任意形状机器人群实验显示，SRL-MPC 在安全性与适应性上显著优于代表性基线，并体现出可扩展性和鲁棒性。

💡 让学习负责**调参和适应**、让优化器负责显式安全，是更稳健的混合控制路线。

🔗 项目链接： https://hanruihua.github.io/srl\_mpc\_project/

🔗 资料来源： https://arxiv.org/pdf/2608.21175

05 · arXiv:2608.21083

🔬 **Teaching is a Process: The TOSS Framework for Modeling Human Teaching Decisions in Human-Interactive Robot Learning**

📌 **Human-Robot Interaction · Interactive Robot Learning · Teaching Models · Open Data**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgua7y3PzKhtGz29szD3QJ6hRrh7xMjBkPLoibdF45XW8q90cYoI6vjtGZOlhtXA6AwE3oFM2k9CS6dMmD8WFx81FRbd7jfuCZXAk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 把人类教学从一次反馈改写为由**触发、目标、信号与策略**共同调节的过程。

📖 有效的人机教学依赖机器人处理需求与人类教学意图之间的对齐，但现有交互式学习往往把教师压缩为被动反馈者。作者开展自下而上的探索研究，让 34 名参与者观察两类机器人强化学习场景，分析早、中、晚阶段共 204 条直觉教学反应。结果显示，教学决策由 Triggers、Objectives、Signals 与 Strategies 构成相互关联的网络，教师还会自然切换教练、工程师或设计者等角色。由此提出 **TOSS Framework**，将教学建模为机器人行为与人类教学动作之间的程序化循环，并开放数据集。

💡 人类反馈的差异未必是噪声，它可能是在表达教师对学习过程的**内部模型**。

🔗 项目链接： https://osf.io/fumd8/?view\_only=9cec60dccbd446f08bd818d0b3612705

🔗 资料来源： https://arxiv.org/pdf/2608.21083

06 · arXiv:2608.21031

🔬 **PhysCaP: Grounding Code-as-Policy Agent with Physics-Informed Exploration**

📌 **Code-as-Policy · Active Perception · Physics-Informed Exploration · Robot Manipulation**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubErQvmmcGibjaDUNqVw6JZHncGVXJ5oYeQXj7hPWjwibF6gMqKnfTdPszOY4njMRSEFhdTAFg1e9ElESBxlqMCuJrLwhia2WAkeU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 让代码策略代理主动试探环境，用本体感觉**无额外传感器估计质量与刚度**。

📖 VLA 策略擅长复现示范，却通常依赖被动观察，难以推断操作所需的隐含物理属性。PhysCaP 在 Code-as-Policy 框架中加入物理信息探索层，通过免训练的物理属性提取模块，仅从机器人本体感觉估计物体质量与刚度。其双代理设计由 Planner 决定何时探索和停止，Prioritizer 过滤不合理交互并按启发式优先级排序，以平衡探索成本与信息收益。在真实桌面操作及 LIBERO 仿真任务中，PhysCaP 相比被动或朴素交互基线，以更少交互和更短执行时间获得可比表现。

💡 主动感知的核心不是多做动作，而是选择**信息增益更高的物理交互**。

🔗 项目链接： https://physcap.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.21031

07 · arXiv:2608.20817

🔬 **GhostTac: Manipulating Tactile Sensors without Physical Contact**

📌 **Tactile Sensing · Physical-Layer Security · EMI Attack · Robot Safety**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYWeMibRyRglOI00R6NYLeR2iaW5Qc1NxZMmNxHzFHibq4k9T0ZRibydXhjZhshzOLpGKiau0YLmK7fJuThaN1hKQ3TmiafFqBgz8HD0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 首次展示针对机器人触觉的**非接触式电磁干扰攻击**，可稳定操纵传感输出。

📖 触觉传感器已成为现代机器人与物理世界交互的关键部件，但其物理层安全长期缺少研究。GhostTac 利用非线性整流和有限带宽放大效应，使精心构造的电磁干扰转化为可绕过板载滤波的持续直流偏移，并通过重塑空间分布和目标位置幅值，实现细粒度、可控的触觉输出操纵。作者在 **10 个传感模块、2 只灵巧手和 15 种不同触觉传感器**上验证攻击，并通过抓取、滑移检测和材料分类三个案例展示其对真实机器人任务的影响。

💡 具身系统的安全边界必须下沉到**传感器物理层**，而不能只防软件攻击。

🔗 项目链接： https://github.com/GhostTac/GhostTac\_CCS · https://ghosttac.github.io/GhostTacCCS.io/

🔗 资料来源： https://arxiv.org/pdf/2608.20817

08 · arXiv:2608.20308

🔬 **DreamHand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery**

📌 **Egocentric Vision · Video Diffusion Model · 3D Hand Recovery · Embodied Data**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguY48Zkmibc0iboSRe1QicZGaOSLqrfaYKJl8VlKJwxGialPk6cpY0Bd4Fk5QsVzWD2Praq3XMEP2luAUgw3yBQ2CqvAWdEX5g0F5U0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

✨ 把视频扩散模型改造成**确定性几何编码器**，恢复遮挡乃至出画双手的连续 3D 轨迹。

📖 第一视角视频可规模化提供具身操作数据，但严重物体遮挡和双手暂时离开视野，使公制 3D 手部轨迹恢复十分困难。DreamHand 不再把视频扩散模型当作依赖多步采样的像素渲染器，而是通过干净潜变量上的单次前向传播提取场景先验，再以双向时空解码器恢复连续双手轨迹；另一配置借助射线相机求解器，可不依赖测试时相机内参。该方法在五个第一视角基准上达到新 SOTA，ARCTIC 与 HOT3D 的 MPJPE-p 分别降低 **30%** 和 **40%**，纳入出画双手后增益达到 **46%—61%**。

💡 视频生成模型的价值不只在生成画面，其潜空间也可成为**遮挡鲁棒的几何先验**。

🔗 项目链接： https://github.com/ggxxii/dreamhand

🔗 资料来源： https://arxiv.org/pdf/2608.20308

**综合观察**

这 8 篇论文的共同变化，是机器人研究不再只追求“把动作做出来”，而是在补齐可部署系统需要的上下游能力：ViTacPhys 与 DreamHand扩展了可用的物理与人体运动信号，Q-Planning 和 PhysCaP 让策略能从部署反馈与主动试探中获益，SRL-MPC 把学习的适应性嵌入显式安全结构，TOSS 与 GhostTac 则分别提醒我们把人类教师和传感器攻击面纳入系统设计。面向更远期的太空采矿，开放资源清单所解决的正是数据、仿真与验证基础设施不足的问题。值得关注的不只是单点指标，而是这些模块能否组合成可验证、可持续改进、对物理风险有感知的完整闭环。
