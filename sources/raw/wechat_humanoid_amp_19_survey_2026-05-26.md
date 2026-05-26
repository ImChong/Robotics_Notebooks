---
title: 万字长文，读懂人形机器人AMP：20篇论文搭起的运动先验圣经
author: 具身智能研究室
date: "2026-05-21 09:00:00"
source: "https://mp.weixin.qq.com/s/YZsm3855iP3TNTTt1aou7w"
---

# 万字长文，读懂人形机器人AMP：20篇论文搭起的运动先验圣经

现阶段运动控制的技术路线逐渐收敛，是时候进行一波总结了，之前的[两万字长文，读懂人形机器人强化学习运动控制：42篇论文搭起的算法圣经](https://mp.weixin.qq.com/s?__biz=Mzg5Mjc3MjA5Nw==&mid=2247495239&idx=1&sn=52b10a8b3b86889cbbc177508273cf3c&scene=21#wechat_redirect)反响不错，但是依旧在一些特定的方向还不够完善，所以才有了AMP这篇。

如果只问一个机器人 **能不能跑起来**，AMP 不是唯一答案。

Mimic也已经证明：只要 reward 设计得足够细、仿真训练足够充分，机器人可以跑得很快，也可以跑得很稳。

但这篇文章真正想讨论的，**不是“谁能让机器人跑起来”，而是另一个更底层的问题**：

机器人跑起来之后，为什么还需要像一个真实身体在运动，而不是像一套被 reward 拼出来的关节轨迹？

这就是 AMP 最有意思的地方。它**不把重点放在“跑步”本身**，而是把跑步、走路、转身、恢复这些动作，重新放回**人类运动分布**里。

换句话说，mimic 可以让机器人照着一段动作跑，RL 可以让机器人为了速度跑，但 AMP 想补的是 **任务完成之后的身体合理性**。

对人形机器人来说，未来的机器人会被**语言模型、VLA、世界模型或者更高层的智能系统调用**，它要在人类环境里走、跑、停、转身、坐下、站起、避让、搬东西、推门、摔倒后恢复。

如果**底层身体动作本身很怪**，上层规划再强，也很难真正落地。

未来具身 AGI 不一定会保留“原始 AMP”这个具体算法形态，但一定绕不开 AMP 的基本思想：让智能体的身体行为受到真实运动分布的约束。

我一直是偏 **技术乐观派** 的人。现阶段具身智能路线还不够清晰，依据现有技术，也很难说已经能直接走到 AGI。世界模型、VLA、运动控制、真实数据、仿真、长时程任务，每一块都有很多待解决的问题，也有可能通往AGI的路，不能发生在现有的这些范式下。

但我不愿意因此低估这些积累。

很多今天看起来很高的技术壁垒，放到足够长的时间里，往往都会被新的数据、新的模型、新的工程系统一点点磨平。

AMP 让我感兴趣，正是因为它把“智能”从符号、视觉和规划，往**身体经验**上推进了一步。它不是 AGI 的单独答案，但它提醒我们：智能体如果最终要进入物理世界，就必须拥有关于**身体运动的经验和约束**。

所以这篇文章不打算把19篇论文硬分成很多类，而是沿着一个问题往下看：

机器人从“能跑”，到“像一个有身体经验的智能体一样跑、走、恢复、交互”，中间到底缺了什么？

01

AMP 解决的不是“能不能跑”，而是“跑得像不像一个身体”

最早的 AMP 来自物理角色动画。

它要解决的问题很现实：如果只给角色一个任务 reward，比如走到目标、越过障碍、击中目标，策略确实可能完成任务，但动作很容易长得奇怪。腿能动，身体也能过去，但整体不像一个自然的运动系统。

这和今天的人形机器人非常像。

人形机器人可以通过速度跟踪学会跑，但跑出来之后，仍然会有一堆问题：膝盖弯曲不自然，手臂摆动怪，躯干姿态僵，脚步节奏不稳定，起停切换不顺，甚至看起来像是在“用关节凑动作”。

AMP 的价值就在这里。

论文 01

AMP：真正的起点不是跟踪，而是分布约束



🔗 **项目链接**：https://xbpeng.github.io/projects/AMP/AMP\_2021.pdf

📄 **论文标题**：AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control

🏫 **机构**：UC Berkeley、上海交通大学

AMP 的核心不是让角色逐帧复现某段参考动作，而是让策略生成的状态转移尽量接近动作数据里的状态转移。

这句话听起来技术味很重，但它背后的含义很直白：

动作不需要一帧一帧照抄，但整体不能偏离人类运动分布太远。

![AMP 原始工作中的风格化物理角色控制](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVZ8ZkBR6JibFre4LKdhIDWOVke2pAjKhibWBF5C7RcXLXdUqC2XyZsLxrSTq4icUTTEh9QeukiaqZv2CdgFeKoXfyEhIyFxWYwvibw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)AMP 原始工作中的风格化物理角色控制

这和普通 mimic 最大的区别就在这里。

Mimic 更像“照着某段动作跑”。AMP 更像“你可以为了任务调整动作，但不要跑出人类运动的合理范围”。

这个差别放到机器人上会更明显。

因为机器人面对的任务不可能永远和参考动作一模一样。**速度会变，目标会变，地形会变，身体状态也会变**。如果策略只能逐帧跟一条轨迹，它很容易被参考动作锁死；但如果完全只追任务 reward，又容易学出一些人类看起来很别扭、部署时也不一定可靠的动作。

所以 AMP 的意义**不是替代 mimic**，而是把 mimic 从逐帧参考动作里松开。它让策略可以更自由地完成任务，同时又不会完全长成 reward hacking 后的怪动作。

这也是我认为 AMP 对人形机器人仍然重要的原因。

人形机器人真正需要的不是一条跑步轨迹，而是一种稳定、自然、可迁移的身体运动分布。

论文 02

ADD：动作自然之外，还要解决 reward 太碎的问题



🔗 **项目链接**：https://xbpeng.github.io/projects/ADD/ADD\_2025.pdf

📄 **论文标题**：Physics-Based Motion Imitation with Adversarial Differential Discriminators

🏫 **机构**：Simon Fraser University、Sony PlayStation、NVIDIA

ADD 可以看成 AMP 作者线里对 reward 工程的一次继续追问。

物理模仿里经常有很多目标：关节姿态、末端位置、身体朝向、速度、接触、稳定性。传统做法通常是把这些 loss 加权求和，然后手动调权重。

问题是，权重一多，训练就变成“调参手艺”。

![ADD 用对抗式差分判别器处理多目标模仿](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVZcwQtuMdL2XpZ3bC5SXXPibOM9MJbkFfF4QiaL7jTx6Unvph8a0I96paEFT2RxbtUrrvF2McTM1mPvBIuNWreqXL8PhR7MOmg8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)ADD 用对抗式差分判别器处理多目标模仿

ADD 的思路是让 discriminator 看当前表现和理想表现之间的差，让它在训练中动态关注哪些目标还没做好。

这件事对人形机器人很贴。

因为人形控制里最痛苦的往往不是没有 reward，而是 reward 太多。要稳、要像人、要省力、要快、脚不能滑、身体不能歪、摔倒要恢复、任务还要成功。

如果每个目标都靠手调权重，系统会越来越脆。

ADD 的价值不是简单把 AMP 换了一个判别器，而是把“哪些地方还不像、哪些目标还没做好”这件事交给训练过程自己去判断。

这对人形机器人很关键。因为走路、跑步、恢复、转身这些动作，在不同阶段真正重要的误差并不一样。起步时也许身体姿态更重要，落脚时接触更重要，恢复时整体状态更重要。如果所有阶段都用一套固定权重，很容易顾此失彼。

所以 ADD 给我的启发是：

AMP 这条线不只是“让动作像人”，它也在试图减少手工 reward 工程。

论文 03

SMP：运动先验应该变成可复用组件



🔗 **项目链接**：https://xbpeng.github.io/projects/SMP/SMP\_2026.pdf

📄 **论文标题**：SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control

🏫 **机构**：Simon Fraser University、Sony Interactive Entertainment、Stanford、Snap、NVIDIA 等

SMP 是我觉得非常值得放进这篇文章的一篇。

传统 adversarial motion prior 往往要和策略一起训练。换一个任务、换一个 controller，先验可能又要重新训练，原始动作数据也要一直留着。

这不太像一个成熟组件。

![SMP 将运动先验做成可复用模块](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEVDYLg9ic6cia60ibicb0EdpJOtmWicWyaRDtoozBobjHuJyaK7Z0W37AQD3EhZZ7Gm611C6gZQzjNXefDXXskx5KZhHiaZdm6na0iaFs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)SMP 将运动先验做成可复用模块

SMP 关心的是：运动先验能不能独立出来，变成一个**可复用的 reward model**。

这其实已经很接近我对未来具身智能的判断了。

未来一个机器人系统里，不应该每个任务都从头学“怎么像一个身体在动”。更合理的方式是，底层已经有一组可复用的身体先验：走路、跑步、恢复、蹲起、转身、交互动作。

上层智能系统调用任务能力时，不需要重新发明身体。

这篇论文的意义就在于，它把 motion prior 从“某一次训练里的技巧”，往“可以被很多任务反复调用的模块”推进了一步。

如果未来机器人系统真的要规模化，身体先验一定不能每换一个任务就重训一次。否则上层的 VLA、语言规划、世界模型再强，底层控制也会一直被重复训练和重复调参拖住。

它只需要在这些身体先验之上做任务组合和目标调整。

论文 04

Kimodo：如果动作数据不够，运动先验也会受限



🔗 **项目链接**：https://xbpeng.github.io/projects/Kimodo/Kimodo\_2026.pdf

📄 **论文标题**：Kimodo: Scaling Controllable Human Motion Generation

🏫 **机构**：NVIDIA

Kimodo 不是机器人控制论文，但它应该放在这里。

因为 AMP 和运动先验最终都依赖动作数据。如果动作数据少、覆盖窄、控制不精确，后面的策略也会被限制住。

![Kimodo 的可控人体动作生成](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVXTUdNXfLYPLswNztdCOKxO1zC8ibqR59Mia2vc3UW8ia52YFH8iaJbuoyFqibWTW174o11nCqibzbVfxrzicPS0FgDoyvduvoiamKHLI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)Kimodo 的可控人体动作生成

Kimodo 做的是大规模可控人体动作生成。它支持文本、关键帧、稀疏关节位置、二维路径等条件。

这件事和人形机器人有什么关系？

我觉得它补的是 **动作来源**。

未来机器人不可能只靠少量 mocap 或真机遥操作覆盖所有身体动作。更可能的路线是：先用大规模人类动作生成模型扩展动作库，再经过重定向、物理过滤、运动先验和 RL 控制器，变成机器人可以执行的能力。

这里有一个容易被忽略的点：AMP 本身并不创造“人类动作经验”，它只是利用这些经验。

所以**动作数据的规模、质量、可控性**，最终会直接决定运动先验的上限。Kimodo 这类工作虽然不直接训练机器人，但它可能会影响后面机器人能学到什么样的身体分布。动作生成越可控，后面的运动先验就越容易覆盖长尾动作，而不是永远停留在少数走跑片段上。

从这个角度看，Kimodo 不是控制器，但它会影响控制器。

动作生成越强，AMP 这类运动先验能学习到的身体分布就越丰富。

论文 05

MotionBricks：生成式动作接口开始接到机器人



🔗 **项目链接**：https://xbpeng.github.io/projects/MotionBricks/MotionBricks\_2026.pdf

📄 **论文标题**：MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives

🏫 **机构**：NVIDIA、ETH Zürich、UT Austin

MotionBricks 比 Kimodo 更靠近机器人。

它关注实时可控动作生成，并且展示了 Unitree G1 上的真实全身控制。

![MotionBricks 将生成式动作接口接到动画和机器人](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXdHDVXK5uQ0CBicbViaI2Eg29u4R3ia3WDE2aFHoaqlrj1vONYiat7ICBMEPrtpchYziaehibuY5nTg7HZnv37eDfbnfE92HibXh9MkY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)MotionBricks 将生成式动作接口接到动画和机器人

这篇论文让我更确定一件事：未来机器人底层动作不会只是单个 policy，也不会只是某个固定轨迹库。

它更可能是一套可以实时组合的动作接口。

MotionBricks 提到 smart primitives，我觉得这个概念很接近未来具身系统里的“身体组件”。

上层 AGI 或 VLA 不可能每次都直接输出底层关节控制。更合理的路线是，上层提出目标，底层通过运动先验、动作生成器、技能模块和控制器，把目标转成自然、稳定、可执行的身体动作。

这也是我觉得 MotionBricks 值得和 AMP 放在一起看的原因。

AMP 解决的是“动作应该落在哪个身体分布里”，MotionBricks 更像是在问“这些身体动作能不能被实时组织和调用”。一个偏训练信号，一个偏动作接口。两者放在一起看，就能看到**从运动先验到身体 API 的方向**。

所以从 AMP 到 MotionBricks，可以看到一条很清楚的演化：

从“让动作像人”，走向“让身体动作成为可复用、可调用、可实时组合的接口”。

02

到了人形机器人，AMP 最先会落在走路和跑步上

回到人形机器人。

我同意一个判断：**AMP 现在最自然、最直接的应用场景，就是走路和跑步。**

因为这是人形机器人最基础的能力，也是最难完全靠手写 reward 调好的能力。

当然，mimic 也能实现跑步。DeepMimic 很早就证明了这一点。很多 motion tracking 方法也能让机器人跑起来。

但问题在于：

跑起来只是第一层，跑得像人、跑得稳定、能从走切到跑、摔倒后还能恢复，是另一层。

这正是 2025 年以后这些人形论文反复讨论 AMP、motion prior、generative prior 的原因。

论文 06

GMP：判别器告诉你像不像，但不一定告诉你哪里不像



🔗 **项目链接**：https://sites.google.com/view/humanoid-gmp

📄 **论文标题**：Natural Humanoid Robot Locomotion with Generative Motion Prior

🏫 **机构**：浙江大学

GMP 严格来说不是传统 AMP，但它非常适合作为对照。

它指出了 AMP 类风格奖励的一个问题：判别器能给出“像不像”的信号，但不一定告诉你具体哪里不像。

![GMP 运动先验框架](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXBl1uwK1ya6WmnPCuBO5kAiaLqskhrFA4zqTeUG4TSLXRKybIkRh81GsIscNAE75sEcAcbY8mhvtjZBKxrYibyt1WSTUlTicL9Vc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)GMP 运动先验框架

人形机器人走路时，很多细节都很微妙。膝盖角度、躯干摆动、脚步节奏、手臂配合，都不是一个简单 reward 就能写清楚。

GMP 试图用生成式运动先验提供**更细的轨迹级指导**。

这就把问题推进了一步。

AMP 类方法常常能告诉策略“你不像人”，但不一定告诉它“你到底哪里不像”。GMP 想补的就是这种更细的指导：不是只给一个风格分，而是让策略更清楚地靠近自然运动轨迹。

这说明一个趋势：

大家不是不需要运动先验，而是开始嫌“只靠判别器”还不够细。

所以我不会把 GMP 看成 AMP 的反面，而是看成同一个问题的另一种回答。

它们都在问：机器人走路时，怎样把人类运动经验放进训练过程里。

如果说 AMP 更像是在训练中画出一条“不要跑偏”的边界，那 GMP 更像是在边界内部继续给出方向感。对人形机器人来说，这种差别很实际：**只知道不像人还不够，策略还需要知道该往哪个身体姿态、哪个步态节奏、哪个轨迹形态去靠近。**

论文 07

ALMI：人形不是一整块，腿和上半身要分开看



🔗 **项目链接**：https://almi-humanoid.github.io

📄 **论文标题**：Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning

🏫 **机构**：中国电信人工智能研究院、上海科技大学、中科大、西北工业大学、清华大学

ALMI 的切入点非常实际：人形机器人的上半身和下半身，在控制里不是同一种角色。

下半身首先要稳定移动，跟随速度指令；上半身则承担动作表达、模仿和未来操作接口。

![ALMI 上下半身对抗式学习框架](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEVFcFdtBafFuTia6p1oZUrfOKVzILfYwdVDiapCHUm750W6mfhvPzpUO1Gp7iaunvQqbY9U6VYc2wqdmQpR2LRRbyaPEQJ1DuHhEE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)ALMI 上下半身对抗式学习框架

如果把全身动作粗暴塞进一个模仿框架里，策略很容易顾此失彼。为了追上半身动作，下半身可能牺牲稳定性；为了保持走路稳定，上半身又可能变得僵。

ALMI 的意义在于，它把这个问题拆开。

我觉得这对 AMP 很重要。

因为 **“像人”不是一个均匀约束**。腿、腰、手臂、躯干在不同任务里优先级不一样。走路和跑步时，腿和躯干的稳定性更重要；交互任务里，上半身自然性和操作关系又更重要。

所以 ALMI 不是简单把 AMP 用到人形机器人上，而是在提醒我们：**运动先验必须理解身体结构。**

如果未来机器人要一边移动一边操作，上半身和下半身很可能会长期处在不同目标下。下半身要稳，上半身要表达意图、保持任务姿态、准备接触。这种分工如果处理不好，策略看起来就会很“整块”，没有人的协调感。

这也说明，未来 AMP 不会是一个全局统一的风格判断。

它一定会更条件化、更分部位、更任务相关。

论文 08

MoRE：复杂地形上，步态不能只有一种



🔗 **项目链接**：https://more-humanoid.github.io/

📄 **论文标题**：MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits Learning on Complex Terrains

🏫 **机构**：中科大、中国电信人工智能研究院、哈尔滨工程大学、上海科技大学

MoRE 很适合说明人形 AMP 的下一步。

它不是只在平地上讨论“走得像不像人”，而是把自然步态放进复杂地形里。

![MoRE 复杂地形步态框架](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXVB9kycO6fYdYThb8sWygQhdWuUAfpRbFpicA1DyYuWicYehVroCdGwicmuGOt1drdIOvWPrOQoy6XyLibibrApMG49NAiaeB71Pxt8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)MoRE 复杂地形步态框架

复杂地形会改变一切。

平地上自然的步态，到了台阶、坡道、沟壑、边缘、室外路面上，可能就不安全。机器人必须根据地形调整落脚、身体高度、重心和速度。

MoRE 的价值在于，它没有把自然步态看成一种固定风格，而是看成多种可切换的运动模式。

这和人的运动更接近。

人不是永远用同一种步态走路。慢走、快走、上坡、下坡、跨台阶、绕障碍，身体都会自然切换。

这篇论文让我觉得，AMP 在人形机器人里不能只服务“视频里好看的一种步态”。

真实环境里最难的不是保持一种漂亮步态，而是**在地形变化时不突兀地切换步态**。平地上的自然性、坡道上的自然性、跨障碍时的自然性，本来就不应该完全一样。MoRE 把这个问题拆成多个 expert，本质上是在让自然性也具备状态依赖。

所以 MoRE 给我的判断是：

AMP 在人形机器人里不能只做平地美化，它必须进入地形和步态切换。

论文 09

Hiking in the Wild：复杂地形里，脚落在哪里比速度更重要



🔗 **项目链接**：https://project-instinct.github.io/hiking-in-the-wild

📄 **论文标题**：Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids

🏫 **机构**：清华大学交叉信息研究院、上海期智研究院、清华大学计算机系

Hiking in the Wild 不是典型 AMP 论文，但它应该放在这条线里。

因为它说明：一旦机器人走到真实复杂地形上，运动先验必须和感知闭环结合。

![Hiking in the Wild 复杂地形展示](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEUo8skY7Nic9NibUAKrBUmmCAgTPBsh9l0DHVlILr0uCWKib7ThnCQpdSsmtq0yMbRQRpoFRP08FpuEdaFxH4yRlVibaw1lL6lLy9o/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)Hiking in the Wild 复杂地形展示

我看这篇时最在意的不是速度，而是落脚风险。

很多失败不是机器人不会迈腿，而是脚踩到了不该踩的位置。边缘、缝隙、台阶外沿、斜坡转折处，这些地方对人来说一眼就能避开，但对策略来说，如果没有合适机制，很容易出事。

这对 AMP 的启发是：

运动先验告诉机器人“人一般怎么走”，但感知必须告诉机器人“现在不能这么走”。

这两者不能互相替代。

这句话其实很重要。

很多时候我们说“像人一样走”，容易忽略**人类走路本身就是感知驱动的**。人不是只靠运动习惯走路，而是一边看地面、一边调整步幅、一边改变重心。真正可部署的人形 AMP，必须和这种感知闭环结合，否则它只能在干净环境里发挥作用。

未来真正有用的 AMP，一定要和视觉、深度、本体感知、落脚安全绑在一起。

论文 10

State-Dependent AMP：走、跑、恢复不该永远靠状态机



🔗 **论文链接**：https://arxiv.org/abs/2605.18611

📄 **论文标题**：Unified Walking, Running, and Recovery for Humanoids via State-Dependent Adversarial Motion Priors

🏫 **机构**：香港大学等

这篇是我认为最贴近“人形 AMP 未来”的论文之一。

它把 walking、running、fall recovery 放进同一个策略里。

![走跑与摔倒恢复统一策略展示](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWAKNOKkjNx6zqKQNWcSVXgGTYZwoSoiatIcOdUicqpUPMZPiaLcVz5yQS4rbhD1DLUibUs4Oicfz5QfLwgPuRwyTeOFiajOMXCOv0oo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)走跑与摔倒恢复统一策略展示

传统做法通常会把走路、跑步、恢复分成不同控制器，再用手写状态机切换。

这当然能做，但边界很麻烦：什么时候算跑？什么时候算摔倒？恢复到什么状态后切回走路？这些逻辑手写起来很脆。

State-Dependent AMP 的意义在于，它让判别器随身体状态变化。

机器人直立时，需要的是走路和跑步的自然性；机器人倒地时，需要的是恢复动作的自然性。

AMP 不应该永远是一个全局风格约束，而应该根据身体状态改变自己约束什么。

这篇论文把“走、跑、恢复”放在一起，我觉得比单独展示跑步更有价值。

因为真实机器人最怕的不是只会跑得快，而是状态一变就失去合理动作。走到跑、跑到摔、摔倒后恢复，这些边界才是真实系统最容易出问题的地方。State-Dependent AMP 正好把运动先验放到这些边界上，而不是只服务单一动作片段。

这句话很重要。

它已经把 AMP 从“像人一样走”推进到了“像一个人一样在不同身体状态之间切换”。

03

AMP 要进入 AGI 路线，必须先变成身体组件

你提到 AGI，我觉得这里可以说得更明确一点。

如果 AGI 只停留在语言、图像、视频层面，它可以不关心 AMP。

但如果 AGI 要进入物理世界，要控制一个身体，要在现实里走、跑、拿、放、推、坐、站、避让、恢复，那么它一定需要某种身体先验。

这个先验未必叫 AMP，也未必使用最早的 adversarial discriminator。

但它一定会继承 AMP 的基本思想：

智能体不能只追任务目标，它的身体行为还必须落在一个合理的运动分布里。

否则，上层智能越强，底层动作越可能暴露问题。

语言模型可以说“跑过去把门打开”，但身体系统必须知道怎么跑、怎么减速、怎么站稳、怎么伸手、怎么接触、失败后怎么恢复。

这不是纯语言问题，也不是纯规划问题。

这是**身体问题**。

论文 11

AHC：多行为蒸馏不是简单拼策略



🔗 **项目链接**：https://ahc-humanoid.github.io

📄 **论文标题**：Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning

🏫 **机构**：哈尔滨工程大学、中国电信 TeleAI、中科大、上海科技大学、哈工大等

AHC 处理的是多行为统一控制。

如果每个动作都训练一个独立策略，机器人部署时会遇到切换问题。走路一个策略，恢复一个策略，爬坡一个策略，跳跃一个策略，最后系统很容易变成一堆拼接模块。

![AHC 多行为蒸馏与强化微调框架](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEUCibQ5aNyRzFw5CE0N9jVuoIXBp7uWpJjx69ia9yqicJQ0IpFjCWqjQSicQ95nGBut1X40OajOEY1yJM3mOPiahkicVSH231y6TuXjE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)AHC 多行为蒸馏与强化微调框架

AHC 通过多行为蒸馏和强化微调，把多个行为压进一个更统一的控制器里。

这类工作和 AMP 的关系，不是“有没有加判别器”这么简单。

它真正指向的是：**未来机器人身体能力必须被统一封装，而不是散落成一堆孤立技能。**

这也是我觉得 AHC 应该放在 AMP 文章里的原因。

如果 AMP 最终要成为具身智能系统的一部分，它不能只是某个单技能训练里的附加奖励，而要**服务更大的身体系统**。多行为蒸馏做的事情，就是把很多身体能力先压到一个更稳定的底层控制空间里。只有这样，上层智能才有可能把身体当成一个相对可靠的接口。

AGI 要调用身体，不可能每次都自己处理策略切换。底层必须先形成稳定的身体系统。

论文 12

HAML：多技能 AMP 的关键是别让条件失效



🔗 **论文链接**：https://www.mdpi.com/2076-0825/15/4/212

📄 **论文标题**：HAML: Humanoid Adversarial Multi-Skill Learning via a Single Policy

🏫 **机构**：山东大学

HAML 也是单策略多技能路线。

它用 conditional adversarial learning，把多种技能放进一个策略里，同时处理条件失效的问题。

![HAML 单策略多技能学习框架](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVp9lYWPRZibkXzsa1tdaUAlJ8pNFA6OCicLDksQ3iaKQktBts2fsFU2uRNnicMjaqibMG1EtrOiaNDj11LNu3T1PQQGClozNEtj5xQc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)HAML 单策略多技能学习框架

这篇论文提醒我一个细节：如果技能多了，判别器不能只判断“像不像人”，还要知道当前到底是哪种技能。

走路的自然性、跑步的自然性、站起的自然性、跳跃的自然性，并不是同一件事。

所以多技能 AMP 的重点不是简单扩大动作数据。

重点是让运动先验理解条件：当前任务、当前身体状态、当前技能意图到底是什么。

这个判断会影响后面很多工作。

如果条件没有被理解，AMP 反而可能给出错误约束。比如恢复动作本来就不像正常走路，跳跃落地也不该像平地行走。多技能场景下，运动先验必须知道**“现在应该像哪一种动作”**，而不是笼统地判断像不像人。

这和未来 AGI 也有关。

AGI 给出的高层目标，本质上也是条件。底层身体先验必须能根据条件改变约束，而不是永远用一个统一“像人”标准。

04

走跑只是开始，AMP 会外溢到任务动作

如果 AMP 只停在走路和跑步，它的想象力有限。

真正值得关注的是：当机器人进入任务时，AMP 的基本思想还在不在？

我认为会在。

因为任务动作同样需要身体分布约束。守门、滑板、坐下、搬箱子、遥操作、多人协作、全身碰撞，这些不是普通 locomotion，但都需要机器人像一个有身体的系统一样运动。

论文 13

Humanoid Goalkeeper：守门任务里，自然反应也是控制目标



🔗 **项目链接**：https://humanoid-goalkeeper.github.io/Goalkeeper/

📄 **论文标题**：Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints

🏫 **机构**：香港大学、上海人工智能实验室

Humanoid Goalkeeper 里，运动先验用于守门任务中的自然全身反应。

机器人不是只要挡住球，还要在快速移动中做出更像人的扑救、下蹲、跳跃和躲避动作。

![Humanoid Goalkeeper 守门动作展示](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEX6uRHzefu6tOIhadsCn4m8EYZvxUVIHMQz3wFuvVXUdsQzYgicIK8Q6iccOicYHNutQtsxia7ru95o5pnGibf2lbC3n1ia2bLxco05o/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)Humanoid Goalkeeper 守门动作展示

这篇论文提醒我：动态任务里，动作自然性不是装饰。

守门需要快速进入姿态。如果动作不自然，身体可能来不及调整；如果只追自然，又可能挡不住球。

真正难的是两者同时成立。

我觉得这篇有意思的地方是，它把 task 和 motion 绑到了一起。

守门不是纯运动模仿，也不是纯任务强化学习。机器人既要根据球的位置调整身体，又要让全身反应保持在合理动作范围里。这里的运动先验更像一种**任务动作约束**：它不是让机器人“表演得像人”，而是让机器人在完成任务时别把身体用坏。

这也是 position-conditioned task-motion constraints 有意思的地方。

同样是扑救，球在左侧、右侧、高处、低处，对应的身体反应不一样。如果只给一个通用的自然性约束，很可能压不住任务差异；如果只追任务成功，又可能学出非常别扭的扑救姿态。它真正讨论的是：**任务位置变化时，身体动作约束也要一起变化。**

论文 14

HUSKY：滑板任务里，AMP 学的是推板经验



🔗 **项目链接**：https://husky-humanoid.github.io/

📄 **论文标题**：HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control

🏫 **机构**：中国电信人工智能研究院、上海交通大学、中科大、上海科技大学、香港大学

HUSKY 里，AMP 被用于学习 human-like pushing motion。

这个任务很特殊：人形机器人和滑板形成一个耦合系统，脚、板、身体倾斜、方向控制都绑在一起。

![HUSKY 人形机器人滑板系统](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWnWn1VfKfVdtK4MCGWK2gCbgfibotm0HefEgIVUTFev04AhZtHveCHL5olVic5N0IbkSQRsiaOswa1lDvibj7jB7RRETaOSiaNgrj8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)HUSKY 人形机器人滑板系统

这里 AMP 不是为了好看，而是为了让推板动作更接近人类可用经验。

滑板任务说明一件事：有些身体经验很难手写。

脚什么时候发力，身体怎么倾斜，什么时候换支撑，什么时候恢复平衡，这些都更像“身体经验”，不是几条 reward 能轻松写完的东西。

这也是 HUSKY 和普通走跑论文不同的地方。

在滑板任务里，机器人不是单独控制自己的身体，还要控制一个会反过来影响身体状态的外部物体。板子的速度、摩擦、方向变化都会改变身体动作。AMP 在这里学到的不是“好看的动作”，而是人在这种耦合系统里的发力习惯和恢复习惯。

所以 HUSKY 让我看到 AMP 的另一个价值：它可以把**人类在特殊运动里的经验**引进机器人控制。

走路和跑步是基础能力，滑板这种任务则更像压力测试。只要外部系统开始反作用于身体，手写 reward 就会变得更困难。AMP 在这里提供的不是万能答案，但它能让策略少走一些反直觉的路。

论文 15

PhysHSI：坐下、躺下、站起，也需要自然身体先验



🔗 **项目链接**：https://why618188.github.io/physhsi

📄 **论文标题**：PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System

🏫 **机构**：上海人工智能实验室、香港科技大学

PhysHSI 把 AMP 用到 humanoid-scene interaction 里：搬箱子、坐下、躺下、站起，以及风格化行走。

![PhysHSI 场景交互任务展示](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXAFrQBhVJSv6PZN5qrMjUFG9IXuoZxUvChNYR83CoXLQicqQbhtQxu9HdPX4AheHBDmG6AJP6ibSy7j53jIh3mu1DwPXQCiasQkE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)PhysHSI 场景交互任务展示

这篇论文非常适合说明 AMP 的外溢。

坐下、躺下、站起听起来很简单，但对人形机器人来说，这是全身接触和姿态变化问题。

如果动作僵硬，机器人可能能完成任务，但看起来不像在人类环境里可接受的动作。

所以我更愿意把 AMP 看成未来服务机器人里的底层审美和安全约束：

它不只追求好看，而是让机器人动作不至于违背人类环境里的身体常识。

这里的**“身体常识”**其实很重要。

一个机器人坐下，如果只是几何上坐到椅子上，并不代表动作可用。它还要控制身体下降的节奏、接触的位置、躯干的姿态，以及起身时的重心转移。PhysHSI 这类工作让我觉得，AMP 的应用范围会从 locomotion 扩展到更日常、更贴近人的场景交互。

这也是服务机器人迟早会遇到的问题。

人类环境不是为了机器人重新设计的。椅子、沙发、桌子、地面、障碍物，都要求机器人理解身体和场景之间的关系。AMP 类先验在这里的作用，不是让动作更“炫”，而是让动作更接近人类环境默认接受的身体方式。

论文 16

CLOT：长时程遥操作里，也需要运动先验兜底



🔗 **论文链接**：https://arxiv.org/abs/2602.15060

📄 **论文标题**：CLOT: Closed-Loop Global Motion Tracking for Whole-Body Humanoid Teleoperation

🏫 **机构**：上海交通大学、上海人工智能实验室

CLOT 关注长时程全身遥操作。

它不是典型 AMP 走跑论文，但里面使用 adversarial motion prior 来抑制不自然行为。

![CLOT 长时程全身遥操作闭环](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUJCgUI12lAdib7PgfjuzfOGPzBjKItOndxXXrfdP0icohBwXC7icib89Vd6peZYuFRojM6npHk0XMEBDHok3Qtd1L7GIhMiaDUE43Y/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)CLOT 长时程全身遥操作闭环

遥操作系统里，人给出的动作意图未必总是机器人可以稳定执行的。长时间执行还会有全局位姿漂移、局部纠偏过猛、动作变形等问题。

这时候运动先验不是用来展示风格，而是用来兜底。

当系统开始跑偏时，运动先验给它一个身体分布上的边界。

长时程遥操作特别能暴露这个问题。

短视频里一个动作做得漂亮，不代表几十秒、几分钟的连续控制也稳定。人的输入、机器人动力学、环境接触之间会不断积累误差。CLOT 里的 adversarial motion prior 更像一个长期约束，防止策略在连续执行中慢慢漂到不自然、不可控的身体状态。

这也说明，运动先验不一定只在训练初期有用。

在长时程闭环里，它更像一种**持续的身体校正机制**。上层遥操作或未来 AGI 给出的意图可能很粗糙，底层控制器需要一边跟随，一边把动作拉回可执行、自然、稳定的范围内。

这对未来 AGI 控制身体也很重要。

上层模型一定会犯错，底层身体系统必须有自己的约束。

论文 17

TeamHOI：多人协作时，不能照搬单人动作数据



🔗 **项目链接**：https://splionar.github.io/TeamHOI

📄 **论文标题**：TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size

🏫 **机构**：Garena、Sea AI Lab、NUS

TeamHOI 做的是多 humanoid 协作搬运。

它用 masked AMP 来处理一个很现实的问题：合作任务里缺少多人体数据，但单人动作数据又不能直接照搬到多人协作。

![TeamHOI 多人形协作交互](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVzibpDOzL85dLtYp9PQhBicWzvXgPOwnPkHddZqGXykjRs6J6jtY52PpydptuXQOvqygM1LMvmGicrj1YmejByf0qPhJ7Ezcwppg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)TeamHOI 多人形协作交互

masked AMP 的思路很有启发。

它不是让所有身体部位都被同一个运动先验管住，而是对不同身体部位做不同处理。

和物体交互的部分更多服从任务 reward，非交互部分继续受运动先验约束。

这其实把 AMP 推到了更现实的一步。

任务交互时，身体某些部位必须为了任务偏离自然动作。比如搬东西时手臂要贴合物体，身体要配合负载，两个机器人之间还要保持协同。如果全身都被同一个运动先验限制，反而会妨碍任务完成。masked AMP 的意思是：**哪里该像人，哪里该让给任务，要分开处理。**

这更像未来 AMP 的真实形态：

局部、条件化、任务相关，而不是一个全局风格分。

论文 18

Deep Whole-body Parkour：复杂动作的本质是身体和环境关系



🔗 **项目链接**：https://project-instinct.github.io/deep-whole-body-parkour

📄 **论文标题**：Deep Whole-body Parkour

🏫 **机构**：清华大学交叉信息研究院、上海期智研究院

Deep Whole-body Parkour 不是 AMP 主线论文，但它对理解人形运动系统很重要。

它把感知接入 whole-body motion tracking，让机器人根据环境几何完成 vaulting、diving、jumping 等复杂动作。

![Deep Whole-body Parkour 全身跑酷任务](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWvXxG7j6DOja6iaJJQLNR4SdvmGBu2hNWgJsnibpZdAa3E1KPZjicOrk9XbIJgMuSREsebQQ6iaq1wRjzZDUNcyXfEy4V9AbZputI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)Deep Whole-body Parkour 全身跑酷任务

这篇论文提醒我们：走跑之外的人形动作，很多都不是单纯姿态问题，而是身体和环境之间的关系问题。

AMP 可以提供自然动作分布，但跑酷这类任务还需要视觉、接触、几何约束和时机判断。

所以 AMP 不能单独成为全部。

它更像身体系统里的一层先验，而不是完整控制系统。

这一点对 AMP 的边界很重要。

如果一篇文章只讲 AMP，容易把运动先验说得太万能。但 Deep Whole-body Parkour 说明，真正复杂的身体能力一定是**多模块合成**的：视觉看环境，控制器处理接触，策略决定时机，运动先验负责把动作拉回合理身体分布。AMP 是关键组件，但不是整个系统。

论文 19

Embrace Collisions：人形不是只能用脚和手接触世界



🔗 **项目链接**：https://project-instinct.github.io

📄 **论文标题**：Embrace Collisions: Humanoid Shadowing for Deployable Contact-Agnostics Motions

🏫 **机构**：清华大学交叉信息研究院、上海期智研究院

Embrace Collisions 的问题意识很直接：

过去很多人形机器人研究默认只有脚和手接触环境。但人类不是这样。

我们会坐下，会躺下，会翻滚，会用身体其他部位和环境接触。

![Embrace Collisions 全身接触动作展示](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXr8RLMnwK1tZfgCedbDjibJI5aZrXwMOXgaLzuEmqdC9zcsE32ibxSEgk7HLPFu6yVAFJ6jsIm7qltFHCuppThpBwEbMhAuY5iaM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=18)Embrace Collisions 全身接触动作展示

这篇论文把“自然身体动作”推到更极端的位置。

如果机器人未来真的要进入真实环境，就不能只会脚底落地和手部操作。躯干、膝盖、手臂、背部都可能参与接触。

这会让传统 locomotion reward 更不够用。

因为接触越复杂，越难手写所有动作细节；但如果完全依赖人类动作先验，又要面对机器人身体和人类身体之间的差异。

所以这篇给我的提醒是：

AMP 的未来不只是让机器人走得像人，而是让机器人学会更完整地使用身体。

我觉得这篇可以作为整条线的一个收束。

如果人形机器人只能用脚走路、用手操作，它其实还没有真正**用完整身体进入世界**。真实环境里的身体能力一定包括摔、滚、坐、靠、支撑、碰撞和恢复。到了这一步，AMP 的问题也会变得更难：它不只是判断步态像不像人，而是判断全身接触中的身体行为是否合理。

05

最后：我会怎么读 AMP 这条线

读完这 19 篇之后，我不会再把 AMP 简单理解成“让机器人动作好看”。

**AMP 是一种把人类运动分布嵌入强化学习控制的思想。** 它可以服务走路，也可以服务跑步；可以服务恢复，也可以服务长时程任务里的身体稳定性。真正重要的不是“对抗训练”这个标签，而是它背后的判断：

智能体的身体行为，不能只由任务目标决定，还必须被真实运动分布约束。

这也是为什么 AMP 会先在人形机器人的走路和跑步里反复出现。走路和跑步太基础了，也太容易暴露问题：能完成速度指令，不代表动作自然；能跑起来，不代表能平稳切换；能保持不摔，不代表摔倒后能用合理方式恢复。AMP 处理的正是这些“任务完成之后仍然不像一个身体”的问题。

但我也不认为未来一定会保留最早 AMP 的算法形态。更可能的情况是，它的思想被拆进更大的系统里：**上游有动作生成和动作接口，中间有运动先验，下游有统一控制和恢复策略，再往上才是 VLA、世界模型和语言规划。**

读完这几篇，再回头看 ADD、Kimodo、MotionBricks、ALMI、HAML、TeamHOI，会更容易明白这条线为什么没有停在“让动作更像人”。

AMP 的未来不只是让机器人更拟人化地走路，而是把人类身体经验变成可复用的底层先验，嵌入到更大的具身智能路线里。

所以这篇文章不只是一个论文清单。

它更像是一张路线图：从能跑，到像人一样跑；从像人一样跑，到能恢复、能交互、能被更高层智能稳定调用。

这也是我作为技术乐观派会持续关注 AMP 的原因。今天看，具身智能离 AGI 还很远，很多路线也还没完全走清楚。但只要机器人最终要进入真实世界，它就绕不开**身体经验、运动分布和底层控制**这些问题。

而 AMP，至少给了我们一条理解“身体先验如何进入智能系统”的清晰线索。

CONTACT

交流与联系

如果你也在关注 **具身智能、人形机器人运动控制、AMP / 强化学习 / VLA** 这些方向，欢迎交流。

• **公众号作者微信**：yzz010329

• **商务微信**：jszn576
