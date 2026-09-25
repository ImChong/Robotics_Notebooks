---
title: 研究了1933篇 IROS 论文，我们看到了机器人学的六项新变化
author: AI科技评论
date: "2026-09-25 10:00:00"
source: "https://mp.weixin.qq.com/s/XvdbbidbJKBszMQwVzvD0A"
---

# 研究了1933篇 IROS 论文，我们看到了机器人学的六项新变化

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/XqAicMdcoiafOdtH3YgeJMl5kKrTA8VyHokTnkgjic8Vv46fxad5HpdqecqRNlxD6AvL1uKoH9oAwD9N9LJRZ3QfdKmUNQSzW75Fx1CVRBX4w4/640?wx_fmt=jpeg&from=appmsg#imgIndex=0)大模型并没有“吃掉”机器人学。

    作者丨马晓宁

    编辑丨岑   峰

****![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/cNFA8C0uVPuQzlZAjS66ZEYbDkdc0DAaC7AvbxPrUmFpdQ5eozPic4tSVCN0YvwfgRX5AIqSHKx8mic2s2a0DjoQ/640?wx_fmt=png&from=appmsg#imgIndex=1)![]()****

9月27日，IROS 2026 将在匹兹堡开幕。重隔31年后，IROS 重回这座钢铁之城。作为全球规模最大的机器人学术会议之一，今年进入正式议程的论文达到 1933 篇。把这近两千篇论文铺开来看，它更像是一张 2026 年机器人研究的横截面：从强化学习、模仿学习，到路径规划、视觉感知、运动控制、灵巧操作，再到人形机器人、触觉和大模型，过去几年机器人领域几乎所有重要技术路线，都同时出现在这张图谱里。但真正值得注意的，并不是 Humanoid、VLA 或 World Model 这些更容易获得产业和媒体关注的关键词。对 1933 篇论文进行多标签梳理后可以看到，Robot Learning / Embodied AI 相关论文约有 809 篇，Navigation / Planning 564 篇，Perception / Vision 556 篇，Control / Dynamics 546 篇，Manipulation 520 篇，而 Humanoid / Legged 相关论文约为 213 篇。由于一篇论文可能同时涉及多个方向，这些数字不能简单相加，但它们说明了一件事：机器人研究并没有因为大模型进入，而被重新归并成一个单一的“AI 问题”。相反，学习、感知、规划、控制和操作正在以前所未有的密度交织在一起。在万众瞩目的 AGI 狂热大背景下，这近两千篇论文却释放了一个信号：**大模型并没有****“****吃掉****”****机器人学。**恰恰相反，在经历了两年的端到端狂奔后，机器人研究正通过三维几何、符号推理和底层控制的‘中间层复兴’，试图在匹兹堡找回那个被像素和 Token 掩盖的物理灵魂。”IROS 2026 真正值得追问的，也许不是“大模型是否正在取代传统机器人学”，而是机器人研究是否正在进入一个更复杂的系统阶段。![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/cNFA8C0uVPsftdZPOcWcnkWugp5Shs2EFib7ZU93GBp4DqVXhrOBCQKQzxcyAIzj1mUKiawHRm6xD4obGgROJm7Q/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

**01**

## ******AI 没有取代传统机器人学，******

## ******而是在重新嵌进去******

如果进一步观察这些论文之间的交叉关系，大模型进入机器人之后真正发生的变化会更加清楚。Robot Learning 与 Manipulation 同时出现的论文约有 276 篇，与 Perception 交叉的有 269 篇，与 Navigation / Planning 交叉的有 225 篇，与 Control / Dynamics 交叉的也有 221 篇。学习方法没有简单挤压掉机器人原有的技术模块，反而越来越深地嵌入这些传统问题之中。这种变化首先体现在规划上。来自石溪大学的 Jorge Mendez-Mendez 在 **《A Systematic Study of Large Language Models for Task and Motion Planning with PDDLStream》** 中，系统测试了大语言模型嵌入传统 Task and Motion Planning 框架之后的表现。研究并没有指向“大模型可以直接取代规划器”，反而显示，几何约束、运动可行性和执行逻辑仍需要传统规划体系参与。这很有代表性。LLM 可以理解“把杯子拿到桌子的另一边”是什么意思，但机器人真正执行时，还必须回答机械臂会不会碰撞、关节是否可达、路径能否执行，以及当前环境有没有发生变化。问题因此从“LLM 能不能替代 Planner”，变成了两者应该如何分工。类似现象也出现在 VLA 的感知端。入围 IROS 2026 Cognitive Robotics 最佳论文候选的 **GeoVLA: Empowering****3D****Representations in Vision-Language-Action Models**，由天津大学、原力灵机和清华大学团队合作完成，作者包括 原力灵机的**谢斌**、清华大学的**石昊**等。它针对的正是现有 VLA 过度依赖二维视觉的问题：传统 VLA 往往从 RGB 图像和语言指令直接生成动作，但真实机器人面对的是三维世界，物体高度、尺寸和相机视角变化，都可能直接影响抓取。GeoVLA 没有继续单纯扩大模型，而是显式加入深度和三维几何信息，把 3D representation 与视觉语言特征共同用于动作生成。换句话说，大模型变强以后，机器人学最传统的“空间几何”又被重新补了回来。控制端也出现了类似现象。来自中佛罗里达大学、印度理工学院坎普尔分校、巴斯大学和马里兰大学帕克分校团队的 **Training Fast Robot Policies with Slow Foundation Models**，研究的是一个很现实的问题：大型 Foundation Model 可以提供更强的知识和泛化，但推理速度太慢，不适合持续运行在机器人高频控制环里。因此，研究者让大型模型更多参与训练，再学习一个更轻、更快的机器人策略负责实际执行。语言模型晚几百毫秒回答问题，用户通常只会觉得它反应稍慢；机器人在抓取、接触或者保持平衡时晚几百毫秒动作，任务可能已经失败。模型能力和实时控制，从来不是同一个问题。更接近物理世界的触觉研究，也呈现出相似方向。来自科罗拉多大学丹佛分校**方新民**等人与肯尼索州立大学团队合作完成的 **DexTact: A Visuo-Tactile Foundation Model for Dexterous Hand Grasping**，把视觉和触觉同时放进灵巧手抓取问题中。它研究的已经不再只是“图像告诉机器人物体在哪里”，而是“手接触物体之后，系统还能感知到什么”。这里，Foundation Model、触觉、灵巧操作和控制被放进同一套系统里理解。因此，2026 年真正值得注意的变化，并不是机器人学正在被 AI 取代。更准确地说，是 AI 开始深入机器人学最传统的环节，同时也在被这些传统问题重新塑形。![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/cNFA8C0uVPsftdZPOcWcnkWugp5Shs2EFib7ZU93GBp4DqVXhrOBCQKQzxcyAIzj1mUKiawHRm6xD4obGgROJm7Q/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

**02**

## ******VLA 从“证明能做”进入“补短板”阶段******

如果说过去两年 VLA 最核心的问题还是“能不能让一个统一模型理解视觉、语言并直接生成机器人动作”，那么到了 IROS 2026，研究重点已经明显向另一个方向移动：模型既然已经能做，接下来要解决的是，怎样让它做得更快、更稳，也更适合真实机器人。在我们的多标签统计中，VLA、VLM、LLM 和 Foundation Model 相关论文约有 162 篇，占全部论文的 8.4%；其中明确涉及 VLA 的论文约有 82 篇。这个规模已经足以形成一条独立研究主线。但它们关注的问题，并没有继续停留在“模型还能不能更大”，而是迅速分化到效率、三维理解、长时记忆、视角变化、现实环境适应和安全等问题上。最直接的例子是 **Shallow-π: Knowledge Distillation for Flow-Based VLAs**。这篇论文来自 42dot、首尔大学和 Samsung Research，并进入 IROS 2026 Best Paper / Best Student Paper Award 候选。它解决的不是如何训练一个更大的 VLA，而恰恰是模型太大、推理太慢的问题，通过知识蒸馏压缩 VLA，让模型更适合真实机器人和边缘设备部署。这暴露了 VLA 进入机器人后的第一个现实约束：模型不仅要“聪明”，还必须足够快。另一个短板是三维空间理解。前面提到的天津大学、原力灵机与清华团队的 GeoVLA，本质上就是在给 VLA 补机器人学最传统的几何能力。视角变化本身甚至已经成为一个独立问题。**AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models** 来自首尔大学与 MIT 团队。它研究的是摄像机位置改变后，原本训练好的 VLA 为什么容易失效，以及能不能在不重新训练整个模型的情况下适应新的 camera viewpoint。真实部署中，相机安装误差、机器人平台差异和视角变化都很常见，一个策略如果只能在固定视角下工作，就很难真正泛化。长时任务则暴露了另一类缺口：模型会执行动作，却不一定会积累经验。**Long-Term Memory for VLA-Based Agents in Open-World Task Execution** 由南京大学、早稻田大学、逐际动力和浙江大学等团队完成，把长期记忆直接接入 VLA Agent，让系统保存已经完成过的轨迹和经验，并在之后重新调用。这里的问题已经不再是一次抓取成功与否，而是机器人能否在长流程任务里记住“前面发生过什么”。来自成均馆大学的 **RoboBRIDGE: A Modular Framework for Bridging Policies to Robust Real-World Robotic Agents** 更进一步。它没有试图重新训练一个更强的 VLA，而是在预训练策略外围加入 Monitor、Perceptor、Planner、Controller 和 Robot Interface。当执行失败时，Monitor 负责发现问题；环境发生变化时，Planner 重新规划；Perceptor 更新场景；Controller 再把高层策略变成实际动作。这个结构很能说明 VLA 的下一阶段正在发生什么。研究开始从“造一个更强的 action predictor”，转向如何围绕它建立一个完整系统。所以，VLA 接下来的竞争可能不只是 scaling。更大的问题正在变成 **system engineering**：一个 VLA 如何与感知、记忆、规划、控制和硬件共同工作，并最终从一个能够输出动作的模型，变成一个能够长期运行的机器人系统。![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/cNFA8C0uVPsftdZPOcWcnkWugp5Shs2EFib7ZU93GBp4DqVXhrOBCQKQzxcyAIzj1mUKiawHRm6xD4obGgROJm7Q/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

**03**

## ******机器人正在重新长出“中间层”******

如果说 VLA 的第一阶段是在尝试把感知、语言和动作统一进一个模型，那么 IROS 2026 另一条越来越清楚的研究线索是：研究者又开始主动在这个端到端链条中间加入新的结构。在1933篇论文中，Reasoning / Memory 相关论文约有 119 篇，占全部论文的 6.2%；其中明确涉及 Reasoning 的有 64 篇，Memory 有 36 篇。它们已经形成一批非常明确的 Session，包括 Agents and Memory for Robust VLA Manipulation、Constraint-Guided Reasoning for Long-Horizon Manipulation、LLM Agents Reasoning Through Robot Tasks、Representing 3D Space for Robot Reasoning。这件事本身就值得追问：如果端到端模型已经可以直接从视觉和语言生成动作，为什么机器人还需要 Reasoning、Memory、Planner，甚至重新引入 Symbolic Constraint 和显式的 3D representation？原因之一，是机器人任务一旦变长，“当前这一帧看到什么”就不再足够。**TempoFit: Plug-And-Play Layer-Wise Temporal KV Memory for Long-Horizon VLA Manipulation** 来自西交利物浦大学团队。它没有重新训练一个更大的 VLA，而是利用模型内部的 temporal KV memory，让现有策略跨时间保存并调用历史信息。但机器人真正需要记住的，也不只是“过去看过什么”。来自东京理科大学团队的 **GaussMemory: Task-Driven 3D Gaussian Scene Memory for Long-Horizon Robotic Manipulation**，进一步把 Memory 放进三维空间。一个物体即使暂时被遮挡，也不能因为摄像头此刻看不到就从机器人的“世界”里消失；一个已经被移动的物体，也需要及时更新位置。因此，机器人里的 Memory 和语言模型里的上下文记忆并不完全一样。它必须和真实三维空间持续对应。推理问题同样如此。**DualCoT-VLA: Visual-Linguistic Chain of Thought Via Parallel Reasoning for Vision-Language-Action Models** 由香港科技大学（广州）与华为团队合作完成。它没有满足于让 VLA 直接输出动作，而是在动作生成前加入视觉 Chain-of-Thought 和语言 Chain-of-Thought：一条处理空间关系，一条处理高层任务逻辑。原本端到端模型试图省掉的“高层—低层”分工，又以另一种形式回来了。入围 Cognitive Robotics 最佳论文候选的 **VL-Nav: A Neuro-Symbolic Approach for Reasoning-Based Vision-Language Navigation**，则主要由纽约州立大学布法罗分校团队完成，第一作者是**杜燚**。VL-Nav 把视觉语言模型的语义理解与显式空间、符号推理结合起来，用于复杂环境中的 vision-language navigation。Symbolic 方法重新出现，并不意味着机器人学回到了完全基于规则的时代。它承担的是大模型目前还不擅长稳定完成的那部分任务：让空间关系、执行条件和约束变得更加明确。从 Memory、Reasoning 到 Geometry、Planner 和 Symbolic Constraint，这些工作表面上采用的技术不同，但都在解决同一个结构性问题：高层语义理解和低层连续动作之间存在巨大的距离。因此，IROS 2026 出现的 Reasoning、Memory、Planner 和 Symbolic Constraint，与其说是在“反端到端”，不如说机器人学正在重新发明大模型与低层控制之间的**中间层**。端到端没有简单失败，模块化也没有简单复辟。更准确的变化是：随着任务从单步抓取走向长时操作，从实验室环境走向开放世界，机器人系统正在重新出现层级。![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/cNFA8C0uVPsftdZPOcWcnkWugp5Shs2EFib7ZU93GBp4DqVXhrOBCQKQzxcyAIzj1mUKiawHRm6xD4obGgROJm7Q/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

**04**

## ******Manipulation******

## ******成为具身智能最密集的战场之一******

如果说 VLA、Reasoning 和 World Model 更多代表机器人“怎么想”，那么 Manipulation 更接近另一个问题：这些能力最终怎么通过一只手、一个夹爪或者一条机械臂，真正作用在物理世界里。Manipulation 相关论文约有 520 篇；进一步筛选 Dexterous Manipulation、Tactile、In-hand Manipulation、Contact-rich Manipulation 等方向后，相关论文约有 225 篇。其中，明确涉及 Tactile 的论文约有 91 篇，明确涉及 Dexterous Manipulation 的有 52 篇。在这 225 篇 Dexterous/Tactile 相关论文中，大约 165 篇同时涉及 Manipulation，79 篇涉及 Learning，57 篇涉及 Perception，52 篇涉及 Control。触觉、学习、感知和控制正在同一批操作任务里不断汇合。IROS 2026 Award Candidate **VTAP Gripper: Synergizing Fingertip Sensing and a Visuo-Tactile Active Palm for Dexterous In-Hand Manipulation**，由普渡大学和哥伦比亚大学团队合作完成，其中包括普渡大学的**胡智娴**，以及哥伦比亚大学的**黄秉豪、李昀烛**等研究者。它没有继续追求一只自由度越来越接近人手的拟人灵巧手，而是设计了三根可重构柔性手指和主动视觉触觉掌面，通过指尖触觉与掌面主动接触完成抓取和手内操作。这里最有意思的是，高水平灵巧操作未必一定依赖极高自由度的拟人结构。机械结构、主动接触和多模态感知之间的协同，同样可以产生复杂操作能力。另一篇机器人机构设计奖候选 **PDS Joint: A Parametric Double-Spiral Joint Tailored for Dexterous Hands**，来自北京理工大学五人团队。他们从更底层的关节结构切入，把柔顺结构、关节刚度、本体感知和学习校准共同放进灵巧手关节设计里。从 VTAP 到 PDS Joint，可以看到硬件本身依然远没有收敛，但硬件也开始越来越难与感知、控制和学习分开。与此同时，触觉正在从“装一个传感器”，走向“进入策略”。**TacVLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation** 由普渡大学和意大利理工学院合作完成，其中普渡团队包括**徐政通、张志远**等研究者。它把触觉直接送进 VLA，在真正发生接触时，让触觉信息参与动作决策。因为在遮挡、精细插拔或者已经建立接触的场景里，视觉提供的信息会快速下降。另一条路线则试图降低部署阶段对触觉硬件的依赖。**HapticVLA: Contact-Rich Manipulation Via Vision-Language-Action Model without Inference-Time Tactile Sensing** 来自斯科尔科沃科学技术学院团队。他们在训练阶段使用触觉信息，让策略学习接触知识，但在推理阶段不再强制依赖实时触觉输入。TacVLA 和 HapticVLA 恰好代表两种不同思路：一种把触觉实时接进策略，另一种利用触觉训练，再尽量把触觉经验压缩到模型里。数据问题也开始浮现。入围 Entertainment and Amusement Paper Award 的 **PianoFingering-1.5K: A Large-Scale Expert Piano Fingering Dataset for Dexterous Robot Learning** 来自中国科学技术大学团队，作者包括**王若宇**等人。他们把专家钢琴指法整理成结构化数据集。钢琴演奏看似特殊，实际却是非常苛刻的灵巧操作任务：多个手指要在极短时间内完成精确、连续、具有强时序约束的接触。把这种专家级动作系统化，本质上是在给机器人灵巧操作提供高质量的人类先验。从硬件、触觉到数据，几条路线最终汇聚到同一个问题：机器人如何形成真正的闭环操作能力。灵巧操作因此正在从“造一只更好的手”，扩展为建立一套**感知—触觉—数据—策略—控制**的完整闭环。![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/cNFA8C0uVPsftdZPOcWcnkWugp5Shs2EFib7ZU93GBp4DqVXhrOBCQKQzxcyAIzj1mUKiawHRm6xD4obGgROJm7Q/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

**05**

## ******Humanoid 从“会走”向“边走边干活”扩展******

人形机器人可能是过去两年产业和媒体关注度最高的机器人形态之一，但如果把它重新放回 IROS 2026 的全部论文版图里，情况会显得更加克制。统计中，Humanoid / Loco-Manipulation 相关论文约有 89 篇，占全部论文的 4.6%。其中，46 篇与 Control 交叉，44 篇与 Learning 交叉，34 篇涉及 Manipulation，32 篇涉及 Planning。真正值得注意的，是这些论文内部的问题正在发生变化。过去几年，人形机器人最重要的技术进展，大量集中在 locomotion；现在已经能看到另一批工作开始把“移动”和“操作”放进同一个系统里。比较典型的是 **ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation**。这篇论文由伊利诺伊大学厄巴纳-香槟分校**何夏麟**等人，联合上海交通大学、清华大学研究者完成，并进入 IROS 2026 Mobile Manipulation 相关最佳论文候选。它不是再训练一个更强的行走策略，而是让人形机器人完成接近物体、拿起、搬运和放下等连续任务。这里最值得注意的不是机器人终于“会搬东西”，而是问题结构发生了变化。在单纯 locomotion 中，核心目标是维持身体稳定；一旦机器人手里拿着物体，手臂动作会改变重心，负载影响步态，脚底接触、上肢运动和抓取任务又必须同时满足。“走路”和“操作”不能再被完全拆开。**SteadyTray: Learning Object Balancing Tasks in Humanoid Tray Transport Via Residual Reinforcement Learning** 来自加州大学圣迭戈分校团队，第一作者为**黄安伦**。它把这种矛盾放大到了一个很直观的任务：让人形机器人端着托盘走路，而且托盘里的物体不能掉下来。SteadyTray 在已有 locomotion policy 上增加 residual policy，专门处理步态给末端带来的扰动。**DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation Via World Model** 则由上海交通大学**殷杰**与清华大学团队合作完成，把视觉和 World Model 引入 whole-body loco-manipulation，用预测模型帮助机器人在长时间视觉控制中维持对状态的理解。VLA 也开始进入同一个问题。**Learning Humanoid Loco-Manipulation with Responsibility-Induced Specialized Experts for Vision-Language-Action Models** 来自首尔大学团队。这项工作针对 locomotion 与 manipulation 动作分布差异很大的问题，引入 specialized experts，让不同专家承担不同控制责任。这再次说明，“通用”不意味着“内部什么都不分”。另一个更极端的例子，是 Award Candidate **Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data**。这项工作由清华大学**张智楷**等人牵头，并联合北京大学、上海人工智能实验室、Galbot 和代尔夫特理工大学研究者完成。打网球要求机器人观察高速运动的球、预测轨迹、移动身体、挥拍，同时维持平衡。它看起来离工厂搬运很远，但暴露的是同一个问题：人形机器人不能只拥有一套稳定步态，然后再简单“加两只手”。真正的 whole-body intelligence 要求腿、躯干和手臂围绕任务共同运动。因此，把 IROS 2026 的人形机器人论文简单概括成“Humanoid 越来越火”，反而会错过真正的变化。更准确的描述是：**研究重心正在从单纯 locomotion，向 whole-body loco-manipulation 扩展。**![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/cNFA8C0uVPsftdZPOcWcnkWugp5Shs2EFib7ZU93GBp4DqVXhrOBCQKQzxcyAIzj1mUKiawHRm6xD4obGgROJm7Q/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

**06**

## ******World Model 很热，但还没有成为主流******

如果只看过去一年机器人领域的讨论热度，World Model 很容易给人一种“已经全面进入主战场”的感觉。但所有论文中，明确涉及 World Model 的论文只有 19 篇，占全部论文约 1%。“热”不等于“多”。真正值得注意的，不是 World Model 已经有多少篇，而是它开始出现在什么位置：四足运动、灵巧操作、精密插接、人形 whole-body control、医疗机器人和导航。Best Paper / Best Student Paper Award Candidate **DAWN: Noise-Robust Quadruped Parkour Via Depth-Denoising World Models** 来自韩国技术教育大学团队。它不是让四足机器人“想象”更真实的视频，而是利用 World Model 对有噪声的深度感知进行建模，帮助机器人在真实环境完成 parkour。这里的 World Model 已经开始成为感知—控制链条中的状态建模工具。在灵巧操作里，**Scaling Cross-Embodiment World Models for Dexterous Manipulation** 由**何子浩**领衔，并联合加州大学圣迭戈分校、MIT 等机构研究者完成。它尝试绕开不同机器人手之间完全不同的关节和动作空间，去学习更一般的“动作如何改变物理世界”的规律。这背后是一个很有意思的假设：未来不同机器人真正能够共享的，也许不是动作本身，而是**动作会怎样改变世界**。**Generalizable Robotic Insertion with World Models** 则由 UC San Diego、NVIDIA、华盛顿大学和南加州大学团队联合完成。它把 World Model 放进精密插接任务。与其让策略死记“这种零件怎么插”，研究者希望模型学习“我的动作会怎样改变当前状态”，再利用预测完成控制。到了人形机器人，前面提到的上海交大**殷杰**与清华团队的 DreamMimic，又让 World Model 进入 whole-body loco-manipulation。而 **RoDyn: Taming Interactive Robot-Dynamic 2.5D World Model for Robotic Manipulation** 则由南洋理工大学与清华大学团队合作完成。RoDyn 试图解决纯二维视频预测的一个明显问题：未来画面看起来合理，并不意味着物理关系正确。它进一步引入深度和机器人动态信息，让预测结果更接近真正可用于机器人控制的状态表示。这恰好指出 World Model 在机器人领域最关键的矛盾。机器人真正需要的，不是视觉上合理的未来，而是**动作条件下、几何上和物理上足够可信的未来**。所以，今年 World Model 的问题不再只是：“如果机器人这样做，未来会长什么样？”而开始变成：“既然我可以预测未来，这个预测能不能真正改变我下一步怎么做？”从这个意义上说，World Model 目前更像是一条正在从**预测世界**向**参与控制**过渡的技术路线，而不是已经统治机器人研究的方法。![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/cNFA8C0uVPsftdZPOcWcnkWugp5Shs2EFib7ZU93GBp4DqVXhrOBCQKQzxcyAIzj1mUKiawHRm6xD4obGgROJm7Q/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

**07**

## ******机器人进入“系统问题时代”******

把这些研究放在一起看，IROS 2026 最值得注意的，并不是某一种技术路线正在“赢”。VLA 没有取代传统规划和控制，World Model 还没有成为主流，Humanoid 也只是整个机器人研究版图中的一部分。真正发生的变化，是这些新方法进入物理世界以后，感知、记忆、规划、控制、触觉、安全和恢复这些问题重新变得不可回避。语言模型过去几年的成功，很大程度上来自把大量任务统一进一个模型。但机器人不同。它不仅要“理解”，还要真正行动，而物理世界会把误差、延迟、接触和失败全部暴露出来。所以，IROS 2026 更像是在回答一个新的问题：当模型已经足够强以后，怎样把这些能力重新拼成一个真正能工作的机器人系统。这并不意味着端到端失败了，也不意味着传统模块重新占据主导。更准确地说，机器人正在重新寻找大模型、规划、控制和硬件之间的边界。未来的竞争，可能不再只是比谁的模型更大、能力更多，而是谁能把这些能力真正稳定地组合起来。**当机器人越来越聪明以后，真正困难的开始变成：怎么让它不出错。**

**为了方便大家抱团交流、互通会议消息，我们专门建了 IROS**2026****参会交流群**！进群你会解锁这些「福利」👇**

- **会议情报站：投稿 DDL、格式规范、评审进度……关键节点第一时间送达，再也不怕错过 deadline，投稿准备稳稳的。**
- **同行茶话会：天南海北的同行都在群里，聊心得、碰思路、攒人脉，科研路上我们同行。**
- **前沿补给站：最新研究成果、热点方向实时同步，结合会议主题帮你捋清投稿脉络，给论文灵感加满油。**
- **现场后援团：（会议期间专属开启）：****到了会场也不用慌——**

  **路线导航**：场馆分布、报告厅怎么走，群里随时问；

  **议题共读**：热门 session、Keynote 探讨，错过也能追；

  **Poster 汇编**：现场海报精华整理，一键收藏不遗漏；

  **约局广场**：约饭局、采访局、交流局……想唠嗑、想合作、想面基，群里发起就成。

**📌 进群传送门：** 扫码进群或添加微信Luyoyo\_2026，备注：ECCV + 单位/学校+姓名+方向。

![Image](https://mmbiz.qpic.cn/mmbiz_png/XqAicMdcoiafPzIa9uaDMlR8xbJGzfPZl2QECRpQogB7JSLIcz34YqsY11yiapOqZJxhmyjjsKjJ8y0CKImsOVqHhOdHgtOsToOGtez7MaicibJg/640?wx_fmt=png&from=appmsg#imgIndex=10)

*搞科研/搞技术，信息差很重要。*

*来，一起快人一步！*

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/XqAicMdcoiafOZnYOCjowITdic8cg9RG8SwlOzblkhQlROsNnxIX6mZ5KAcIUAK0va4rQ84pDoficS7v86pZibxcGqqSqTwhial7S7YicdebVKI85Q/640?wx_fmt=jpeg&from=appmsg#imgIndex=11)

**上车，带你看遍全球 AI 顶会精华**

可独家畅览：

专家演讲PPT

大会报告全文

热门论文解读

学术新星访谈

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/XqAicMdcoiafM7wHXNibq2Azg96Yujmunrpqj6c6ha1iaeWUkln8z6Uhia2S8A7kibExUCH37MyMzAensbYib1lAsNVfib2XicYVRPDic4EVhcL90aE4c/640?wx_fmt=png&from=appmsg#imgIndex=12)

扫描上方二维码

或点击「阅读原文」关注专区。

//
推荐阅读
[![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/XqAicMdcoiafOl75PEn5PpwBuMDv1iaYBETLXLvYBht2ibrkOh0sfrlW4WdUIV3dNMtAziaOP9cJEQoTlSjH4iajFicrO6S21pIZgdFYabgEMa1OZc/640?wx_fmt=jpeg#imgIndex=13)](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247746177&idx=1&sn=4425aabd9307dd19fe9e580477327261&scene=21#wechat_redirect)

[从三维视觉到世界模型：空间智能为何成为 AI 走向物理世界的共同主线？| ECCV 2026复盘](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247746177&idx=1&sn=4425aabd9307dd19fe9e580477327261&scene=21#wechat_redirect)


[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/XqAicMdcoiafNX0SmFGbbibaDmk7qxB5JzTTSXlSvr5Eiafq7pIpkRUt3VumxjibKgZhp5icT8en4MyDRCr4j47SGq4cXHBj8M7XEG2KarYBR6Zn4/640?wx_fmt=jpeg#imgIndex=14)](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247745897&idx=1&sn=c752cd8a5f7cbad2d73e7ae6a91fe54c&scene=21#wechat_redirect)

[朱玉可 ECCV 2026 演讲：大模型走过的三级跳，机器人正在重演 | ECCV 2026](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247745897&idx=1&sn=c752cd8a5f7cbad2d73e7ae6a91fe54c&scene=21#wechat_redirect)

# 未经「AI科技评论」授权，严禁以任何方式在网页、论坛、社区进行转载！ 公众号转载请先在「AI科技评论」后台留言取得授权，转载时需标注来源并插入本公众号名片。
