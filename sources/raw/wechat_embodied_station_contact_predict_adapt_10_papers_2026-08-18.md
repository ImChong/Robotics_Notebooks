---
title: 具身智能下一站：这10篇机器人论文，把“看懂”推进到“会接触、会预测、会适应”
author: 具身智能小站
date: "2026-08-18 09:00:00"
source: "https://mp.weixin.qq.com/s/IxmKI4_JYy1KBfp_JCZFLw"
---

# 具身智能下一站：这10篇机器人论文，把“看懂”推进到“会接触、会预测、会适应”

点击下方卡片，关注**【具身智能小站】**公众号

---

📅 2026年8月18日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 10 篇近期**具身智能**与机器人论文，内容覆盖**触觉/力觉**机器人学习、自动化优化器、**人机交互预测**、**布料分拣**、**技能迁移**、**双臂灵巧抓取**、人群跟随、视频**世界模型**、社交机器人**持续学习**和**动作监督**视觉注意力。整体来看关注一个核心问题：机器人如何从“看懂场景”进一步走向“稳健接触、可控预测、长期适应”。

这 10 篇论文的主线不是单点模型变大，而是把物理交互、约束优化、时序预测和社会上下文接入机器人学习闭环，让机器人在真实环境里更可控、更可解释、更能迁移。

01 · arXiv:2608.07558

🔬 **Learning Physical Interaction: A Survey of Tactile- and Force-aware Robot Learning**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZ5bPYYwxUNRt8FMufS4MM9b2SVOODcxMfzibspoiarb2BR5MQuUMIbdWRfvfw5KCicONab4o7AjS4dmibBor0UkqZS99tfju7clX4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 **Embodied AI · **Tactile/Force-aware** Learning · **Survey****

✨ 触觉与力觉从补充传感器，升级为物理交互主线。

📖 这篇综述从物理交互出发，指出接触敏感操作不仅依赖视觉感知和动作生成，还依赖**力调节**与**自适应控制**。论文提出 **TF-ART** taxonomy，把触觉、力觉、视觉、语言和本体感知，以及多阶段系统中的高层策略、动作细化和底层控制，放进统一层级框架中梳理。

💡 接触任务的关键不只是看见，而是会调力。

🔗 项目链接： https://github.com/NTUMARS/Awesome-Tactile-Force-aware-Robot-Learning

02 · arXiv:2608.07539

🔬 **AutoPSO: A Metaframework for Automated Particle Swarm Optimization**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYJ99nbMBgzhh2SVBf2m8EE5yQz0a62R5HCPLJ06uh0ptAYf8glb25SZKCuAN3AFrgqfdVaGyDBtEARDINQZJqc0T4saSwu9LY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 **Auto Optimization · **PSO** · Neuroevolution Robotic Control**

✨ 把手工设计 **PSO** 变体，改造成可搜索、可复用的自动化流程。

📖 论文指出传统 **PSO** 变体多依赖问题特定的人工设计，跨任务泛化弱，且主流实现受 CPU 约束。**Auto**PSO**** 将 **PSO** 优化建模为**双层过程**：外层搜索有效组件组合，内层实例化候选变体解决目标任务并反馈；同时借助 **EvoX** 的种群张量化和批量评估，在数值基准和**神经进化机器人控制**任务中发现更强的 **PSO** 变体。

💡 机器人控制里的优化器，也在走自动化设计路线。

🔗 项目链接： https://github.com/EMI-Group/AutoPSO

03 · arXiv:2608.11051

🔬 **HUI360: A 360° Egocentric Dataset and Baselines for Human-Robot Interaction Anticipation**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYo0Fn2xPlIzQ8JwNKkibRomwzedR4mzgSaWnk4deWSxqkXE18ZcbH7QqkiazwbT4cZoVDkGNF9KSxsp8DKqpNqvkGicRjJfvDricQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 ****HRI** Anticipation · **360°** **Egocentric Dataset** · **Benchmark****

✨ 从移动机器人第一视角采集野外 **360°** 人机互动，补上主动交互预测数据缺口。

📖 面向人群环境中的主动机器人，论文把自动预测人机互动意图定义为**具身智能**的重要感知问题。**HUI360** 是面向野外**人机交互预测**的大规模数据集，由移动机器人在三个月内、多个环境和多天采集，覆盖自然自发的人类行为；论文同时发布自动标注流水线、**1M** 预处理标注、研究用途的原始全景图像和基线评测，并提供**跨数据集评估**。

💡 社交机器人要主动，先得看懂人是否会靠近。

🔗 项目链接： https://hucebot.github.io/hui360/

04 · arXiv:2608.10648

🔬 **Precise Top-Layer Fabric Segmentation for Fabric Destacking with Edge- and Shape-Aware Deep Networks**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguakphtDACJGicfiby4MtFylZorkuFGNvbV05GjqaB7gdoFXo6XHTwTMBP9euo9gxHtbqG2NDmAmAWQZzI2NYAHD5FodmlymaUUGw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 **Robotic Manipulation · **Fabric Destacking** · **Segmentation****

✨ 面向**布料分拣**的顶层区域识别，把边界和形状同时纳入监督。

📖 **布料分拣**需要精确分割最上层布料，但层间边界细微、外观相似，常让语义或边缘分割方法失效。论文提出面向顶层布料分割的训练架构，在 encoder-decoder 框架上加入 **edge-aware branch** 和 **shape-aware branch**，前者强化边界，后者利用来自 **CAD** 模型的参考 mask 对齐整体形状；真实布料数据集实验显示该方法优于已有基线。

💡 软物体操作常输在感知边界，而不只是抓取策略。

🔗 项目链接： https://github.com/bhattner143/top-layer-fab-seg

05 · arXiv:2608.10600

🔬 **BooST: Bridging Semantics and Motions for Efficient Skill Transfer**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZetM5safLTJ8lRicOlSicNnDUQjF1P97QqpYxosuqQaySBsb41X2KQNGURUJJSqeSZukQibk47VxnoC6tkaSe45AdITCia0Yd25dI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 **Skill Transfer · Cross-modal Representation · **Policy Distillation****

✨ 同时编码做什么和怎么动，提升机器人少样本**技能迁移**。

📖 论文聚焦技能抽象：可复用的时序扩展行为要能跨任务、跨域泛化，并在真实机器人上保持鲁棒和高效。现有方法往往只捕获高层语义意图或底层运动动态，导致下游适应需要大量域内数据。**BooST** 采用两阶段框架，先用 **cross-modal VQ-VAE** 统一编码语义意图与运动动态，再蒸馏成轻量策略；仿真和真实机器人实验显示其在**少样本适应**、跨域**技能迁移**和**视觉动态干扰**鲁棒性上更优。

💡 可迁移技能需要语义和运动共享表示空间。

🔗 项目链接： https://boost-robots.github.io/

06 · arXiv:2608.10383

🔬 **Real-World Cooperative Bimanual Dexterous Grasp of Large Objects from Single-View Observations**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYkl4kJypAXqLlYvHONDYR2iaYYfYSacLgSHC7y7l5g0EIHiadVf56v6cMWliaI7Lz7597LXjviapnN1OAolWMniaslXhUX2yToGw5E/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 ****Bimanual** Dexterous Grasp · **DDPM** · **Force Sensing****

✨ **单视角**也能生成**双臂灵巧抓取**，并在真实机器人上在线细化。

📖 **双臂灵巧抓取**大物体是机器人操作中的难题，既缺完整 3D 模型，又难生成物理可行的协作抓取动作。论文提出真实世界双臂抓取框架，包括含关节角、视觉观测和力信号的**多模态**数据集，基于 **DDPM** 的关节级抓取配置生成模块，以及结合运动规划与**在线抓取细化**的执行策略，从**单视角**输入生成可执行抓取并降低对完整 3D 模型的依赖。

💡 双臂协作要同时解决几何不完整与接触稳定。

🔗 项目链接： https://github.com/zhangdana483/real\_bi\_dex\_grasp/

07 · arXiv:2608.10056

🔬 **Navigating the Proximity-Safety Balance: Constraint Decomposition for Human Following in Pedestrian Crowds**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaATG2wgaNiapo6POFbuClO9klw79ygYr0s4gR6HsHYLqMwRLOYrhD73GHhGQ10ccF2sHv3qnP9ClDTeBBqxeTKDWLBADTg4QnI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 ****Human Following** · Multi-constraint RL · **Crowd Navigation****

✨ 把跟紧和安全拆成可调约束，而不是塞进一个 reward。

📖 在人群中跟随目标人类，本质上存在距离接近与安全避障的冲突。论文指出，现有强化学习方法常把二者压进单一 dense reward，导致权衡隐式且难调。该方法将人类跟随分解为稀疏任务奖励和独立成本约束，用有直接行为含义的阈值管理各类约束，并把人类运动**预测不确定性**纳入 RL cost；仿真、分布外测试和真实机器人部署验证了其 **proximity-safety balance**。

💡 跟随机器人需要可解释的安全旋钮，而不只是更高回报。

🔗 项目链接： https://nav-ps-balance.github.io/

08 · arXiv:2608.13489

🔬 **DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgua07LBBlFZKnvM9jnRBvWFyVW6mtnHtbb2zj7DXvIWx9YScJKN1bvJvYbSZSmyKkYWvv7cpy5MlZ3geChzMNhs2ZXa9oeJlU8w/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 **Video World Model · **Action Conditioning** · Robotic Manipulation**

✨ **世界模型**不只要像视频，还要忠实响应机器人动作。

📖 **DreamX-Phi 1.0** 面向机器人操作中的动作条件视频**世界模型**：给定观测帧、语言指令和由末端位姿与夹爪状态组成的动作序列，预测未来观测。论文强调真实感并不等于动作忠实性，因此把每只机械臂的 **SE(3)** 变换通过 **PRoPE**-style 几何编码注入 attention，并加入轻量 depth branch、**SAM3** masks 与冻结 **V-JEPA** teacher 来维护几何和小物体一致性，最后通过 **distribution-matching distillation** 提升部署效率。

💡 机器人**世界模型**的核心指标正在从真实感转向可控性。

🔗 项目链接： https://github.com/AMAP-ML/DreamX-Phi

09 · arXiv:2608.13448

🔬 **Mind the Context: Continual Learning of Socially Appropriate Robot Actions via Environmental-Social Disentanglement![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZU7pMAkdCPnxA4FuznyQMiboqJxCKwT9fjSicrxUZH2gzcWZAdrDeVJyvAoNlygkDClGh8VdBD74TTdibO5wYdP9vICrounsuZ4s/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)**



📌 ****Social Robot** · Continual Learning · **Environmental-Social Disentanglement****

✨ 社交机器人**持续学习**时，把环境线索和社会线索拆开建模。

📖 社交机器人进入不同环境时，相似空间布局可能对应完全不同的合适动作，且这些规范无法预先穷举。论文面向 **domain-incremental** continual learning，指出环境线索和社会线索会共同决定清洁、服务、发起对话等动作是否合适。**EDD** framework 通过双分支显式拆分 environmental 与 social-agent knowledge，并用 **replay-based rehearsal** 缓解遗忘；实验显示其优于多种**持续学习**基线。

💡 社会适当性不是单场景分类，而是跨场景记忆问题。

🔗 项目链接： https://github.com/Cambridge-AFAR/Mind-the-Context.git

10 · arXiv:2608.13422

🔬 **Attention from Action, for Action: Emergent Visual Bottlenecks for Policy Learning**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubvHpIMS0f8AicVPPL0Lbe5IZfusPD9ZC7xiaFtQUxnfWpDcUScUDu53mpcTfBTLHknmFleePhmZKVr9LlD8TdXr0CALpCKzNEkE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

📌 **Visuomotor Policy · **ROI** · Action-supervised Attention**

✨ 不靠人工 **ROI** 标签，用**动作监督**学会该看哪里。

📖 视觉 bottleneck 可以通过 **ROI** 聚焦策略输入，提高数据效率，但常依赖外部空间标签，或用固定的末端执行器 crop 作为动作派生启发式，容易在任务阶段变化时错位。**Seeker** 是一种 task- and state-conditioned readout，从冻结 **DINO** 特征出发，用聚合视觉证据迭代更新 query，仅通过**动作监督**产生 **progress-aware **ROI****；该 **ROI** 可用于 RGB crop、mask-guided background augmentation 和 **point-cloud filtering**，并在仿真与真实机器人中提升数据效率和鲁棒性。

💡 示教动作本身就能提供控制相关的视觉注意力监督。

🔗 项目链接： https://github.com/zheyu-zhuang/seeker

**综合观察**

这组论文释放的信号很清楚：**具身智能**正在从视觉-语言驱动的任务理解，继续向**触觉/力觉**、**世界模型**、可调约束、安全导航、社会上下文和**动作监督**注意力延伸。真正值得关注的不是某个单一模块，而是这些模块如何被放进机器人闭环：感知要能服务接触，预测要忠实于动作，策略要能迁移到真实硬件，安全与社会适当性要能被显式控制。

**资料来源**

01. https://arxiv.org/pdf/2608.07558

02. https://arxiv.org/pdf/2608.07539

03. https://arxiv.org/pdf/2608.11051

04. https://arxiv.org/pdf/2608.10648

05. https://arxiv.org/pdf/2608.10600

06. https://arxiv.org/pdf/2608.10383

07. https://arxiv.org/pdf/2608.10056

08. https://arxiv.org/pdf/2608.13489

09. https://arxiv.org/pdf/2608.13448

10. https://arxiv.org/pdf/2608.13422
