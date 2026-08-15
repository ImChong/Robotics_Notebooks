---
title: "机器人论文密集上新：从 Ego2Robot 到 MANGO-Grasp，下一轮竞争焦点变了"
author: 具身智能小站
date: "2026-08-11 09:10:00"
source: "https://mp.weixin.qq.com/s/nKF7rxH-OuJz68galP3Xpg"
---

# 机器人论文密集上新：从 Ego2Robot 到 MANGO-Grasp，下一轮竞争焦点变了

点击下方卡片，关注**【具身智能小站】**公众号

---

📅 2026年8月10日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 9 篇近期**具身智能**与**机器人**论文，覆盖**人形机器人动作先验**、**机器人书法**、**手部符号化**、**操作世界模型**、**ego-to-robot数据合成**、**四足搜救**、**action chunking**、**VLA指令泛化**与**跨手型灵巧抓取**。整体来看，这批工作不再只拼模型规模，而是在追问：机器人怎样把数据、表征和控制结构做得更可迁移。

从 pose geometry、physical brush、anatomical unit、embodied latent，到 semantic re-binding 与 morpho-kinematic descriptors，作者都在把“可泛化”写进机器人表征和控制接口。

01 · arXiv:2608.03227

🔬 **PFM-HR: Pose Flow Matching for Humanoid Robots**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZ9cGOl9hKS2yDDesYT3HAmgWdE7FWSxy1CclVWfle3Z4j0qVOIQqEFgObrvKia2zakTsOlPFFPSFuTPER6Tl9XKmaRq6vT1ekY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 **Humanoid Control · Motion Prior**

✨ **Pose Flow Matching**把无序 pose 数据变成可复用的人形动作先验。

📖 这篇工作针对 physics-based humanoid tracking 中动作先验的两类短板：时间先验依赖有序 motion clips，普通 pose prior 又难以约束策略 rollout 中的姿态转移。作者提出 **PFM-HR**，直接在大规模无序 pose 数据上训练 flow matching prior，并用 **Pose Geometry Score** 衡量 rollout 中关节坐标变化是否贴合先验捕捉到的局部 pose 几何。PGS 被用于调制 tracking reward，从而在冻结先验的同时引导探索更结构化的人形动作变化。

💡 人形控制的重点正在从“跟得上”转向“动作变化是否像人”。

🔗 项目链接： https://github.com/gaoyukang33/PFM-HR

02 · arXiv:2608.03198

🔬 **Bridging Online and Offline Handwriting via Differentiable Physical Rendering**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubSH9t4l6mSjCbRSYKUt6pAh9cp9DueCNJnmxPcCVGumvII8ibibDiauGCFoR7avricAYwXb6kRUaODZmLKLEjpIdQ5lLbLBiceJgUc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 **Robotic Calligraphy · Differentiable Rendering**

✨ 用**可微物理笔刷**把笔迹轨迹和离线手写图像接到同一个框架里。

📖 现实手写生成长期分成 online trajectory 与 offline image 两条路线：前者保留结构和时序，后者更像真实图像但丢掉 stroke order。论文指出难点在于缺少把 stroke kinematics 与 pixel-level appearance 连接起来的显式物理模型，也缺少配对 trajectory-image 数据。作者提出 compact physical brush model 与 differentiable brush renderer，并组成统一 online-offline handwriting generation framework，覆盖 text-to-stroke、brush parameter observer、renderer 和 zero-shot image refiner，最终用实验与真实**机器人书法**演示验证结构和视觉保真度。

💡 这类工作把“生成图像”重新拉回到可执行的物理轨迹。

🔗 项目链接： https://seonmip.github.io/onoff/

03 · arXiv:2608.03127

🔬 **DigitCode: Symbolic Tokenization of Hand Motion by Anatomical Units**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZhRTgoKeJzMaRdZLMahswmfeMg9pzj0nfL1mNL4YcmPicBNrtIc67xjN2HteBamdGv3HKHMX0ibzhsEdZ5VKR8CBPw8tV0maT4k/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 **Hand Motion · Symbolic Tokenization**

✨ **DigitCode**把手部动作 token 的粒度问题落到 bone、finger、whole hand 的解剖层级。

📖 手部动作承载最细粒度的人类活动信息，但当前 hand generation、understanding 与 robot learning 多依赖连续 joint angles 或 MANO 参数，准确却缺少可索引、可编辑、可验证的结构。论文从 Hand Labanotation 出发，追问一个更底层的问题：符号应覆盖哪个 anatomical unit。**DigitCode**沿手部 unit hierarchy 适配、分组并分层 HL alphabet，把符号表示的量化误差降低约四分之三；同时，per-finger token 可作为 training-free editable handle，用于修复异常生成手和机器人重定向。

💡 手的表示不只是精度问题，还是能否被编辑和迁移的问题。

🔗 项目链接： https://digitcode-demo.github.io/

04 · arXiv:2608.02990

🔬 **EmbodiedVAE: Disentangled Video VAE for Efficient and Controllable Embodied Manipulation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZ9diantvu9ibGXrTSjXvPq6usCAia8DmMyWXVyt20uaibpMTrPao8Ed9ne15EA5yBE73Bz2ibrduxTMzAsgibtrLeKBwfCib6ibN5RQco/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 **World Model · Video VAE**

✨ **EmbodiedVAE**为机器人操作世界模型重做 video VAE 的 latent 空间。

📖 Latent diffusion models 已经推动 embodied manipulation world models，但现有 LDM 常沿用面向自然场景优化的 VAE，忽略机器人操作场景里“机器人运动”和“背景环境”的差异，导致 latent 不够紧凑也不够可控。论文提出 **EmbodiedVAE**，用 dual-encoder single-decoder 架构和 asymmetric spatio-temporal compression 自动解耦 robot arm motion 与 background，并引入 optimal-transport-based consistency module 维护运动 fidelity 和帧间一致性。实验显示它在高压缩率下提升重建质量，并支持更精细 action control。

💡 世界模型能否控得住，往往先取决于 VAE latent 怎么切。

🔗 项目链接： https://github.com/Mutual-Luo/EmbodiedVAE

05 · arXiv:2608.02580

🔬 **Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubr14bquhpw3Y46bb0lrMibEoFkWqh3Dibn67xl5wf7ENzcorwWOyfnlJg9g9UuSKDmh84csWCSjR9ribDr8w3nWUQZGQPddeksIg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 **Robot Data · Ego-to-Robot Synthesis**

✨ **Ego2Robot**把第一视角人类操作视频扩展成大规模机器人训练数据。

📖 通用机器人操作策略需要大规模、多样化 demonstrations，而 egocentric human manipulation videos 天然包含场景和任务多样性。论文提出 **Ego2Robot** pipeline，通过 action retargeting、robot-arm visual synthesis 和 multi-level quality curation，将 curated datasets 与 in-the-wild videos 转成机器人训练数据。该数据覆盖 15 种机器人形态、18,561 小时训练量；作者还扩展 RoboTwin2.0，用视觉外观、场景布局、embodiment morphology 和任务语义等扰动轴评估泛化，并在真实机器人部署中验证联合预训练收益。

💡 机器人数据瓶颈的一个现实出口，是把人类视频变成可训练的机器人视角。

🔗 项目链接： https://www-ye.github.io/ego2robot\_blog/

06 · arXiv:2608.02571

🔬 **Situation Aware Frontier Prioritization for Quadruped Search and Rescue**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguY6cvVWlroM9GzFxgheZBVUUy17cuAxlDs8ZTzffic1VvZiayiadb4eAQngUMYicCXVgc7FKdKJ7WGLib2ic92vvFublvHLM5xJ8MiaI8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 **Quadruped · Search and Rescue**

✨ 四足搜救探索不只看扩图，还要把**救援相关性**纳入 frontier 排序。

📖 四足机器人适合进入轮式系统受限的室内灾害环境，但未知救援场景中的探索不只是扩展地图，还要平衡发现受困者的概率。论文提出 situation aware frontier prioritization，用经典 frontier exploration 框架为基础，在 frontier ranking 中加入 information gain、observation deficit、rescue relevance、terrain penalty 与 travel cost。作者在 Gazebo 中用四足机器人测试两个室内救援场景：简单场景下方法差异不大，复杂 clutter 与 frontier ambiguity 增强后，该方法取得最高 completion rate 和 victim recovery。

💡 搜救机器人需要的不是更贪心地探索，而是更懂任务价值地探索。

🔗 项目链接： https://github.com/ricardoGrando/

07 · arXiv:2608.02547

🔬 **Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYr2tPOHzSE0SnP0AQM5dn3sQTwM8mDtPsJxz5d9qeVAourBcrSaDARyrzLlxcCTyKsfXZsf8ia50oNusic6O4BRzcRwnrMJrMTA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 **Behavioral Cloning · Action Chunking**

✨ 论文把**action chunking**的收益拆成非马尔可夫表达、误差传播和隐式 ensemble。

📖 Action chunking 已是机器人 behavioral cloning 的关键组件，但它为什么有效并不清楚。论文通过模拟和真实机器人实验检验常见解释，认为 temporal consistency、horizon reduction 与 representation learning 都不足以解释其成功。作者发现 chunking 的部分收益来自更强 non-Markovian expressivity 和更低 compounding error，在不少设置中 delayed policies 也能捕捉这些效果；更额外的收益来自**implicit ensembling**，即 action-chunked policies 学到多种 temporal relationships，从而表现得像模型 ensemble，提升鲁棒性和泛化。

💡 把 chunking 当工程技巧不够，它实际在改变策略类的表达方式。

🔗 项目链接： https://action-chunking.github.io/

08 · arXiv:2608.02497

🔬 **Grounded Semantic Re-Binding for Robust Instruction Generalization in Vision-Language-Action Models**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguY8Jh5hibxntyOv3c0KcicHFmK8MWbkttF2IK5PxS5ZIwfYlduT75xAtD6okeuR9dRs5mBcHDwnTy8x7icR3svv4JibHWsvnGXzq50/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 **VLA · Instruction Generalization**

✨ **GSR**指出 VLA 改写指令崩溃的关键，不是语义不懂，而是 joint routing 脆弱。

📖 VLA 在机器人操作中表现突出，但 canonical instructions 一旦被 paraphrase，性能可能灾难性下降。论文的 probing 发现，当前 VLA 内部仍保留正确 task identity，失败更可能来自 dynamic visual observations 与 text 的 joint encoding 引入系统性 feature shifts，而下游 action policy 对这些变化极其敏感。作者提出 **Grounded Semantic Re-binding**，显式融合独立抽取的 task semantics 与 native visual features，并从零训练重新初始化的 action expert。GSR 在 LIBERO-Para 上最高提升 44.6% success rate，并进一步提出 0.33B 参数的 **ParaVLA**。

💡 VLA 鲁棒性不一定靠堆数据，结构解耦本身就是杠杆。

🔗 项目链接： https://github.com/AutoLab-SAI-SJTU/GSR-ParaVLA

09 · arXiv:2608.02014

🔬 **MANGO-Grasp: Mahalanobis Fields over Geometry-Oriented 3D Gaussians for Cross-Embodiment Dexterous Grasping**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguapYk5ciaanwGBM97f6VjBuKz3rpM8Lic4EGOdicqlhEK7ZJp0OCJuhlOSRJ363mgw3ZF9ze9JC8oJF7gUrkTvzLOd4XW9DN6CGhc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

📌 **Dexterous Grasping · Cross-Embodiment**

✨ **MANGO-Grasp**用 geometry-oriented 3D Gaussians 和 morpho-kinematic descriptors 做跨手型抓取。

📖 跨本体灵巧抓取希望在不同多指手之间合成稳定抓取，并减少 embodiment-specific tuning。论文认为现有 interaction-centric 方法虽有效，但 object representation 对局部表面几何表达不足，robot descriptor 也没有同时显式编码形态和运动学。**MANGO-Grasp**把物体表示为 geometry-oriented 3D Gaussian primitives，把机器人手表示为 surface keypoints 与 morpho-kinematic descriptors，并用 keypoint-primitive pair 上的 Mahalanobis fields 作为训练目标和推理优化指导。论文报告其在仿真和未见过的 SharpaWave hand 上均有提升，并实现 86% 真实实验成功率。

💡 跨手型泛化需要把接触方向性、物体几何和手的运动学同时显式化。

🔗 项目链接： https://connor-zh.github.io/MANGO-Grasp/

**综合观察**

综合看，这批论文最值得关注的不是单点指标，而是**接口意识**变强了：PFM-HR 把人形动作约束落到 pose 流形，Ego2Robot 把人类第一视角数据转成机器人训练资产，GSR/ParaVLA 把语言语义从脆弱 joint routing 里拆出来，MANGO-Grasp 则把形态和运动学显式编码进跨手型抓取。对具身智能圈内人来说，这说明下一阶段的竞争不会只在“大模型更大”，而在**数据来源**、**表征粒度**和**控制闭环**能否真正跨场景、跨任务、跨本体复用。

**资料来源**

01. https://arxiv.org/pdf/2608.03227

02. https://arxiv.org/pdf/2608.03198

03. https://arxiv.org/pdf/2608.03127

04. https://arxiv.org/pdf/2608.02990

05. https://arxiv.org/pdf/2608.02580

06. https://arxiv.org/pdf/2608.02571

07. https://arxiv.org/pdf/2608.02547

08. https://arxiv.org/pdf/2608.02497

09. https://arxiv.org/pdf/2608.02014
