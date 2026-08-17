---
title: 机器人论文又卷到哪了？9篇新作看懂具身智能的下一步
author: 具身智能小站
date: "2026-08-17 09:00:00"
source: "https://mp.weixin.qq.com/s/UsgswMgDw4Kdpt5qI9fxnA"
---

# 机器人论文又卷到哪了？9篇新作看懂具身智能的下一步

点击下方卡片，关注**【具身智能小站】**公众号

---

📅 2026年8月17日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 9 篇近期机器人与**具身智能**论文，内容覆盖操作策略加速、自然语言任务计划、**World Action Models** 的语义/时空增强、自动驾驶鲁棒感知、长序列拆解规划、**社交导航**、手术内窥镜控制和视觉强化学习。整体来看一个问题：机器人系统如何把“看见、想象、规划”更稳定地变成可执行动作。

本期主线不是单纯堆大模型，而是把**未来预测**、**语言语义**、**空间轨迹**、速度/安全成本和部署约束接到**控制闭环**里，让策略更快、更稳、更**可验证**。

01 · arXiv:2608.09138

🔬 **SpeedTuning: Speeding Up Policy Execution with Lightweight Reinforcement Learning**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaSxSO1dRlcaSZGKkC5Fjl97lTapcaCKUKRqCcVvEcfD20gMcdpMEpiaJx01JEzsO8yqZbRtBKSJ5g1YKqicMrpo9aL2YdBgiapuc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 **Robot Manipulation · Lightweight RL**

✨ 不用重新采数据，只给模仿策略学一个速度倍率。

📖 学习型机器人策略在**真实部署**中常被**执行速度**拖住：模仿学习受硬件约束和示教者速度影响，固定倍速插值又容易牺牲成功率。SPEEDTUNING 提出一个轻量强化学习框架，为基座策略的动作预测最优速度倍率，不需要额外采集数据；论文报告其在倒、抛、取等动态和精细任务上实现超过 2.4x 的**速度提升**，并保持相对充分的成功率。

💡 速度可以成为策略之外的独立控制维度。

🔗 项目链接： https://daivdyuan.github.io/speed-tuning/

02 · arXiv:2608.08884

🔬 **SHRIMP: Iterative Refinement of Robot Task Plans**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZQVV4yicck61aF6nVXUGGEK1AhVSnUrY4hpk9ciceSbMbR1eqqFu0lXfk8NB2ibAibhvMp9WZapuwTq9BAiaib1wUqa6JQtNvM67ZP0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 **Human-Robot Interaction · Task Planning**

✨ 把自然语言计划放进仿真里反复改，再上真机。

📖 协作机器人进入制造、农业和医疗等场景后，普通用户仍难以编程或调整行为。自然语言和 LLM 可以降低门槛，但语言任务描述存在语义歧义，生成式模型也缺少让用户验证计划如何转成动作的透明度。SHRIMP 提供一个仿真驱动的人在环操作规划界面，用自然语言生成层级 primitive plan，并允许用户通过重新提示和显式纠错反复修改；用户满意后先在仿真中验证，再执行到真实机器人。N=35 的用户研究显示，它提升了用户感知控制感和机器人透明度。

💡 **可验证**的交互界面比一次性生成计划更接近**真实部署**。

🔗 项目链接： https://wisc-hci.github.io/SHRIMP/

03 · arXiv:2608.08839

🔬 **SG-WAM: Text-Grounded and Spatial-aware Semantic Guidance for World-Action Models**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZyLqsPvZ0FEiavibDnR2NMzeiaYZbaSRraOjLuTLOs1cDDicavAiaKavcKyVjsqwMoU7c99NwicDMngGq3ia7mSzbcBXBFvItxicFYxDU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 **World Action Model · Semantic Guidance**

✨ 给 WAM 加一个 VLM 语义规划器，让视频预测听懂指令。

📖 World-Action Models 已成为机器人操作里的一个重要范式，但许多 WAM 主要依赖视觉线索生成未来视频和动作，而现成文本编码器通常独立于视觉观察嵌入指令，导致预测视频和语言指令语义错位，进而影响动作准确性。SG-WAM 使用 VLM 作为语义规划器，预测 text-grounded 与 spatial-aware 的 **semantic foresight**：前者锁定正确目标物体，后者提供场景几何，再作为高层**语义引导**注入 WAM。论文称仿真与真实实验展示了更精确的操作和更强的指令跟随能力。

💡 WAM 的短板正在从“会不会生成”转向“是否语义对齐”。

🔗 项目链接： https://livfour.github.io/SG-WAM/

04 · arXiv:2608.08815

🔬 **Distilling Vision-Language Models for Robust Traffic Sign Perception in Autonomous Vehicles**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguY8TWuPCEWSNictTXlWU52KiceUd1wxEFcapb1WEJvKChDt3yZSCvjpjdCdu7Lv6GtQNbh3xXENYpTxicRaNkX7dL8LVe5YTHllgc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 **Autonomous Driving · VLM Distillation**

✨ 把 VLM 的语言原型蒸馏进交通标志识别，推理时不加负担。

📖 交通标志识别模型在干净数据上表现强，但面对阴影扰动、自然光干扰和打印补丁等物理可实现攻击时仍很脆弱，已有防御还可能只对单一攻击有效并损害干净精度。LAMDA 将语言 grounding 结构转入 TSR 训练：用冻结 OpenCLIP 文本编码器，从 VLM 生成的标志描述和类别名构建两个固定 prototype bank，并通过两个辅助损失监督视觉特征。推理时 adapter 和 prototype bank 被丢弃，只留下标准 backbone 和分类器。论文报告其在 GTSRB、LISA、四种 backbone 和三类物理攻击下均提升**鲁棒性**，阴影攻击最高 +12.5pp，自然光攻击最高 +13.2pp，并基本保持或改善干净精度。

💡 语言先验在车端感知里更像训练约束，而不是部署负担。

🔗 项目链接： https://github.com/pedram-mohajer/LAMDA

05 · arXiv:2608.08773

🔬 **PEEL: Parallel Extraction for Long-Horizon Disassembly Planning via Scale-Invariant Sampling**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguaHLNV69gKgJ89qlGn9nUgUicpFRFkibIic1R4ot9icN9Bib8iauRkJloyjpK6yOicVpF8aPFHINnk9ZGxAtWF4eGknDJnmwGuRRjULT0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 **Motion Planning · Disassembly**

✨ 长序列拆解不是只排顺序，还要在窄缝里找可执行移除路径。

📖 多部件长程拆解要求机器人在狭窄逃逸通道里计算一串**无碰撞**移除动作。PEEL 面向这类任务，用采样式运动规划求解单个部件的移除路径，并通过 scale-invariant sampling 在 burn-in 阶段估计物体尺度，再用方向采样器利用该尺度信息。该采样方案被集成进 MAB-RRT，规划器根据奖励信号在不同采样器间切换；PEEL 并行运行一批规划器，得到部件移除顺序图。论文报告 MAB-RRT 在 76 个装配体单部件拆解上达到 100% 成功率，并用 Fetch 机器人求解了 4 个包含 10 到 17 个部件的长程拆解问题。

💡 拆解规划的难点是顺序、尺度和碰撞自由运动同时成立。

🔗 项目链接： https://peel-disassembly.surge.sh/#code

06 · arXiv:2608.08323

🔬 **MPPI Planning with Gaussian-Based Human Cost Function for Social Navigation**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZicD51FjVfUK1m2Hoiatx1RFk7msO0veib3nKn0EqibNKdXOciciafPKIcG9CkK2dMU8UGHz9bxibUIbIcozGIs5aBp71TPOErooiaN84/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 **Social Navigation · MPC**

✨ 把行人未来运动写进 MPPI 成本，而不是把人当静态点障碍。

📖 拥挤空间中的安全导航需要规划机器人“人将会在哪里”，而不只是“人现在在哪里”。MPPI 是有效的采样式规划器，但不少实现把行人编码为当前位置的静态点障碍，在动态场景中低估风险。PGIF 将行人预测沿整个规划 horizon 前向传播，并编码成与行人运动方向对齐的各向异性高斯排斥场；场的前向扩展随行人速度增长，形成 motion cone danger zone。该公式闭式且可跨 rollout 并行，论文称没有可测计算开销；在 300 个随机人群场景中，PGIF-MPPI 在各密度下碰撞率为 0%，而 vanilla MPPI 最高达到 82%。

💡 **社交导航**里的安全成本必须面向未来，而不是只看当前帧。

🔗 项目链接： https://github.com/ChinmayMundane/PGIF\_MPPI

07 · arXiv:2608.08023

🔬 **4D-WAM: Infusing Spatiotemporal Awareness into **World Action Models** through Trajectory Fields**

![User attachment](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZnGvYxQDD8yzBKgAs9qAjAOcpicHG0p4eQzXo6uFDSFuf0A9EJwdibatrBhmeTP7AwTsl5DqDnFX8XOfCwYj3UAsNPE2U0WFSBc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 **World Action Model · Trajectory Fields**

✨ 把 3D **轨迹场**灌进 WAM，让模型学到局部运动和远期目的地。

📖 WAM 同时建模视频预测和动作生成，但通常仍在 2D 像素空间表示视频，与机器人动作执行所在的 3D 空间存在表示鸿沟；近期 3D 方法虽引入几何信息，却没有充分利用 3D 结构的动态。4D-WAM 是一个 model-agnostic 训练策略，通过 representation alignment 把 3D trajectory fields 中的时空知识注入 WAM。它包含两个互补目标：motion alignment 对齐相邻帧特征变化，推动局部 4D awareness；destination alignment 通过注意力式相似分布差异，引导模型从源帧推断最终目的地。论文称该策略在空间理解、执行精度、**鲁棒性**、泛化和通用性上带来提升。

💡 轨迹级 4D 表示可能成为 WAM 接近物理执行空间的桥。

🔗 项目链接： https://github.com/lishanyqy/4DWAM

08 · arXiv:2608.07876

🔬 **SurgLAT: Surgical Latent Attention Tracking for Depth-Aware Robotic Laparoscope Control**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguasacRUU13EC5oAFcL6hyd0Ng9BkB3icqibqLVFFHiagEj867Q9L2BeUl8DeVoCiadjo7yvaoM8GV51MyA0W3jm5iaTfn4hwoRtI2qM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 **Surgical Robotics · Latent Attention**

✨ 把术者关注区域建成随时间演化的隐状态，驱动腹腔镜视野控制。

📖 自主腹腔镜相机控制需要持续理解术者在动态手术场景中的操作意图，而目标操作区域不是稳定物体，而是一个随时间演化的 latent attention state。SurgLAT 是一个因果在线框架，用于隐式手术注意力建模和自主腹腔镜视野控制：它使用冻结 DINOv3 编码器和 state-conditioned spatial token mixer，在 memory-guided spatial prior 下提取操作证据；selective causal latent memory 同时建模短期运动连续性和长期手术意图演化。该 latent state 被解码为概率注意力热图和操作区域，并结合 RCM 约束控制与 redundancy-aware null-space initialization 部署到机器人平台。论文在真实手术视频和物理腹腔镜平台上验证了遮挡、快速运动和目标切换下的在线跟踪与稳定视野调整。

💡 **手术机器人**更需要建模“意图轨迹”，不是只做目标检测。

🔗 项目链接： https://surglat-home-page.pages.dev/

09 · arXiv:2608.07870

🔬 **V-Simba: Unleashing the Architectural Potential of RL in Visual Continuous Control**

![User attachment](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgub7qEBBjN4YiaXFs5Upiajv9nNOQ6rjqpZYqUWOicicw3JXF4pxMBf1WA7BwqiaO9lN3J7a4TUJJt4dah17FWYcfHajfG3gjnLxLgBM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

📌 **Visual RL · Architecture Design**

✨ 视觉 RL 不只靠新算法，网络结构本身也能提升**样本效率**。

📖 **样本效率**仍是强化学习在真实机器人中落地的核心挑战，因为采集数据代价高；在视觉 RL 中，高维输入还会遮蔽学习信号。以往视觉 RL 更多关注动力学模型或探索策略等算法解法，而 state-based RL 的新进展显示，架构设计本身也能显著提升**样本效率**。V-Simba 受 Simba 架构启发，构建在带数据增强的 SAC 之上，通过加入 normalization layer 稳定训练，并用 pointwise convolution 降低计算。论文报告 V-Simba 在 DMC、Adroit 和 Meta-World 上匹配或超过当前方法，同时比 DrQ-v2 更具计算效率。

💡 在视觉控制里，稳定训练的架构选择本身就是算法贡献。

🔗 项目链接： https://github.com/DAVIAN-Robotics/V-Simba

**综合观察**

综合看，WAM 方向正在从“生成好看的未来”转向“生成对动作有用的未来”；与此同时，SpeedTuning、PGIF-MPPI、PEEL 和 SurgLAT 这类工作强调执行端的速度、安全和几何约束。对**具身智能**落地来说，下一阶段关键不只是模型规模，而是世界模型、VLM 语义、MPC/采样规划和真实机器人部署之间的接口是否足够干净。

**资料来源**

01. https://arxiv.org/pdf/2608.09138

02. https://arxiv.org/pdf/2608.08884

03. https://arxiv.org/pdf/2608.08839

04. https://arxiv.org/pdf/2608.08815

05. https://arxiv.org/pdf/2608.08773

06. https://arxiv.org/pdf/2608.08323

07. https://arxiv.org/pdf/2608.08023

08. https://arxiv.org/pdf/2608.07876

09. https://arxiv.org/pdf/2608.07870
