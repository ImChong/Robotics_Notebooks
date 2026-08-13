---
title: 强化学习必备知识②：机器人运动控制完整pipeline
author: 深蓝具身智能
date: "2026-06-25 10:56:00"
source: "https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247505497&idx=1&sn=0f63d89762a07ba7ac642d876bfba5eb"
---

# 强化学习必备知识②：机器人运动控制完整pipeline

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/uwFbeBKoFGezvgBL1gXhorOr071NjWVhIoWJFhcU7Otk8t40whTFgJVzNX2KibotwdPlJtk38SZ6CN7ibravFJnap1vkDps4yScSTJpibjrqg0/640?wx_fmt=gif&from=appmsg#imgIndex=1)

弄懂这套强化学习流程，上手工程实战项目

> 大家好，这里是【深蓝具身智能】。
>
> 本文出自我们公众号开设的新专栏——《具身智能基础》。
>
> 这是本栏目下的第六篇文章。在上篇《强化学习必备知识①》中，我们用50行代码跑通一个强化学习具身控制的最小闭环。本文延续《强化学习》系列分享，来梳理【机器人强化学习运动控制的技术链路】，全文近 5000 字，建议收藏阅读。

---

[💙](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)[订阅《具身智能基础》专栏](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)

你的订阅和收藏，将支持我们把这件事持续做下去✨

强化学习在机器人运动控制领域的重要性已是共识，就不再这里过多赘述了。但这套范式对应的完整工程落地逻辑却鲜有梳理。

这种从“人工设计规则”到“数据驱动学习”的转变，究竟是怎么做到的？

今天这篇分享，我们依旧回归技术底层，系统梳理一下机器人强化学习运动控制的技术管线（以四足机器人为例）。

涵盖基础闭环、分层控制、PPO算法、特权信息蒸馏、奖励函数设计、域随机化及GPU并行仿真等核心模块。

**我们开设此账号，除了想要向各位对【具身智能】感兴趣的人传递前沿权威的知识讯息外，也想和大家一起见证它到底是泡沫还是又一场热浪？****欢迎关注****【深蓝具身智能】**👇

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=2)

机器人是如何“学习”的？

## 要理解机器人强化学习，我们首先要搞懂一个基础框架：

## 强化学习闭环。

举个例子，你可以把强化学习想象成训练一只小狗，你不会告诉小狗“先抬左前腿 30 度，再收缩右后腿肌肉”，而是给它一个指令，当它做对了，你就给一块肉干（奖励）；做错了，你就轻轻拍它一下（惩罚）。

在机器人领域，这个过程被抽象成了四个核心要素：




1. 观测（Observation）：智能体传感器采集的局部状态观测，包含关节角度、角速度、机身倾角、地形高度图、相机视觉等，对应 POMDP 局部观测空间；

2. 动作（Action）：策略网络输出的控制指令，具身场景多为连续动作空间；

3. 环境E（Environment）：物理仿真引擎 / 真实物理系统，依据机器人动作求解动力学方程，输出下一时刻观测；

4. 即时奖励（Reward）：基于任务指标量化的单步收益，用于指导策略迭代优化。



![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/uwFbeBKoFGdJBXxZBBH7Miat5OujPlRELYmQ0OOmmFf2HTJq32Xn496YZm6aP2PanW5csPj8grGianW4CiabwOU0iawkDoe8wfuItpqXb3qXsjM/640?wx_fmt=jpeg&from=appmsg#imgIndex=3)

▲图1 | 强化学习的基础闭环：智能体（Agent）通过观察环境状态，采取动作，并根据环境反馈的奖励来不断优化自己的策略©【深蓝具身智能】编译



智能体核心载体为深度神经网络参数化策略，训练目标为最大化折扣累积回报，为折扣因子。

完整训练过程为循环执行交互采样、梯度更新，通过大量试错迭代收敛至最优策略

——这个过程，就是策略（Policy）的训练。


```bash
# 单步强化学习基础交互流程
obs = env.reset()  # 初始化观测o_0
for t in range(max_timesteps):
    action = policy_net(obs)  # 策略网络输出动作a_t
    next_obs, reward, done, info = env.step(action)  # 环境交互，获取o_{t+1}, r_t
    buffer.append((obs, action, reward, next_obs, done))  # 存储轨迹样本
    obs = next_obs
    if done:
        obs = env.reset()
# 基于样本池更新策略网络参数θ
update_policy(buffer)
```


▲强化学习最底层的单智能体环境交互
![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=4)

## 机器人强化学习的完整管线

在实际工程中，仅仅有一个基础闭环是无法适配真实机器人动力学非线性、执行器延迟、硬件扰动等问题的。

为了让训练出来的策略既聪明又稳定，研究人员通常会采用分层控制的架构。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGeTCIoHyib786pbvSHrSISsCWF1Dj3rrFbkRSq7FibJmVibDlGIVKiaeh4Vyd2Dmw7TeY7BQIGYujTblnibUR5wkhbPr8Zgljd7mHw4/640?wx_fmt=png&from=appmsg#imgIndex=5)

▲图2 | 机器人强化学习的典型训练管线：高层神经网络输出目标动作，低层控制器负责具体执行，物理引擎提供环境反馈©【深蓝具身智能】编译

从这张图中我们可以看到，整个系统通常分为两层：

- 高层：DRL 策略网络（大脑）



输入传感器观测，输出目标关节位置



这是通过强化学习训练出来的神经网络。它的运行频率通常是 50 Hz（每秒做 50 次决策）。

它根据传感器传来的数据，思考一下，然后给出一个宏观的指令，比如：“所有关节移动到这个新的目标角度”。

- 低层：PD 控制器（小脑/脊髓）

如果让神经网络直接控制电机输出多大的力矩（扭矩），不仅训练难度极高，而且一旦遇到突发干扰，机器人很容易失控摔倒。

因此，主流的做法是让神经网络只输出“目标位置”，然后交给底层的 PD 控制器（比例-微分控制器）去执行。标准 PD 力矩公式：

qt为当前关节角度，为关节角速度；比例增益、微分增益为控制器固定参数。

PD 控制器运行频率极高（通常在 200 Hz 到 1000 Hz 之间）。它就像机器人的脊髓反射一样，实时对比“当前关节角度”和“目标关节角度”，然后计算出电机需要输出多大的力矩。

这种“高层算位置，低层算力矩”的模式有三大优势：

- 降低训练难度：神经网络不需要去学习复杂的底层物理力学，只管定目标就行。
- 自带柔顺性：PD 控制器可以像弹簧一样，吸收足端落地时的冲击力，保护硬件。
- 弥合虚实鸿沟：真实电机和仿真模型总有差异，底层的 PD 控制器可以帮神经网络兜底，抹平这些微小的硬件差异。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=6)

## PPO 算法：为什么大家都在用它？

分层架构确定训练管线后，需选择稳定的策略优化算法完成网络迭代。

如果你去看机器人强化学习的论文，十有八九会看到一个名字：PPO（Proximal Policy Optimization，近端策略优化）。

在 PPO 出现之前，强化学习训练机器人就像“走钢丝”。如果策略更新的步子迈得太大，好不容易学会的一点点走路技巧可能瞬间崩溃。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGfiaiah53ISAxs0bHBmQnNyricOu8IOEkwp1HxQCKZia0pTV4CFiaLz5raektst7uLLqrxAuT3Kl76C54octalsgibVHQuybutM9tlZM/640?wx_fmt=png&from=appmsg#imgIndex=7)

▲图3 | PPO 算法的核心网络结构：通过限制新旧策略的差异，保证训练过程的稳定收敛©【深蓝具身智能】编译

PPO 的核心能力在于它的“截断机制（Clipping）”。约束新旧策略动作概率比值，限制单次梯度更新幅度，标准 Clip 目标函数：

其中：


：新旧策略概率比；


：GAE 广义优势函数，衡量当前动作相对均值的收益；


：截断阈值，工程常规取值 0.2。


```powershell
初始化策略参数 θ_0，值函数参数 φ_0
for k = 0, 1, 2, ... do
    用当前策略 π_k = π(θ_k) 与环境交互，收集轨迹集合 D_k = {τ_i}
    计算折扣回报 R̂_i = Σ_{k=i}^{H} γ^{k-i} r_k
    基于当前值函数 V_{φ_k} 计算优势估计 Â
    通过最大化PPO-Clip目标更新策略：
        θ_{k+1} = argmax_θ (1/|D_k|) Σ_{τ∈D_k} Σ_i min( r_i(θ)Â_i, clip(r_i(θ), 1-ε, 1+ε)Â_i )
    通过最小化均方误差拟合值函数：
        φ_{k+1} = argmin_φ (1/|D_k|) Σ_{τ∈D_k} Σ_i (V_φ(s_i) - R̂_i)²
end for
```


▲一轮又一轮「采集仿真交互轨迹→计算回报与优势→更新策略网络→更新价值网络」的标准训练流程

简单来说，PPO 在每次更新神经网络时，都会把“新策略”和“旧策略”做个对比。

如果它发现新策略相比旧策略变化太剧烈，它就会强行把更新幅度“截断”，限制在一个安全的范围内。

这就好比教练在教运动员，每次只允许纠正一点点动作细节，绝不允许推翻重来。

这种“小步快跑、稳扎稳打”的机制，使得 PPO 极其稳定，成为了目前机器人领域最主流的算法。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=8)

## 特权信息与 Teacher-Student 架构

PPO 解决仿真内训练稳定性，但在仿真环境里，机器人可以说是开了“上帝视角”：

它可以轻易知道地形的确切高度、地面的摩擦系数、甚至自己重心的精确位置。

我们把这些真实世界中很难获取的数据，称为特权信息（Privileged Information）。

但在真实世界里，机器人只能靠自己身上的相机和惯性传感器（IMU），不仅视野有限，数据还充满了噪点。

怎么解决这个矛盾？

研究人员发明了 Teacher-Student（师生蒸馏） 架构。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/uwFbeBKoFGd2knzTZd0Vzj0Rs5Qzewlp9BAfKWy7zH96FHenXjnuoMibevialwehez0WOibdCdLu7KMYnufmw3fxCngafJ5XeWibSMYGuibUhaxA/640?wx_fmt=jpeg&from=appmsg#imgIndex=9)

▲图4 | 经典的 Teacher-Student 训练架构：先利用特权信息训练一个全知全能的老师，再让只能获取普通传感器数据的学生去模仿老师的动作©【深蓝具身智能】编译

整个过程分为两步：

- 阶段 1：教师网络训练





  输入完整特权信息与基础观测，在仿真环境中训练全局最优运动策略，具备完整环境动力学感知能力。

- 阶段 2：学生网络蒸馏训练







  仅输入真机可用局部观测，以最小化师生策略 KL 散度为蒸馏损失，同时叠加任务奖励损失，联合优化学生网络参数：

  为 PPO 原始强化学习损失，为蒸馏损失权重系数。

  通过蒸馏，学生网络可仅依靠局部传感器观测，推理等效全局动力学特征，复刻教师稳定运动行为。

算法与蒸馏架构确定后，策略迭代方向完全由奖励函数约束，奖励函数是定义机器人运动行为的核心约束模块。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=10)

## 奖励函数：“调教”机器人

如果说算法是引擎，那么奖励函数（Reward Function）就是方向盘。

机器人最终走成什么样，全看你奖励什么、惩罚什么。

以四足机器人为例，早期研究中，工程师们为了让机器人走得像狗，会把狗的动作录下来，强迫机器人去模仿（也就是模仿学习）。

通常只设定几个最基础的奖励和惩罚：

其中：

速度跟踪奖励：如果你让你往前走 1m/s，你走到了，就给你加分。

能耗惩罚：电机输出的力矩越大，扣分越多（逼着机器人学会省电）。

动作平滑惩罚：如果关节动作一抖一抖的，扣分（逼着机器人动作顺滑）；

姿态惩罚：如果身体倾斜太大，扣分；

为各分项权重超参。




就靠这几个简单的规则，神经网络为了拿到最高分，会在成千上万次的试错中，自动涌现出类似动物的“小跑（Trot）”步态。

因为物理规律决定了，对于四条腿的结构来说，小跑就是最省力、最稳定的移动方式。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=11)

## 跨越虚实鸿沟：域随机化

**无论奖励函数如何精细，仿真模型与真实物理世界之间固有的动力学偏差仍是部署阶段的核心障碍：**

电机有延迟，地面有坑洼，甚至机器人的某条腿可能比图纸上重了 100 克。这种仿真与现实的差异，被称为 Sim-to-Real Gap（虚实鸿沟）。

为了跨越这条鸿沟，最常用的就是：域随机化（Domain Randomization）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGeF11YBS3j77GCCH8WOgbYb023mYwNibTic0sFrrVWRlzicfwPuap0bib4f53AIGzApYkiaE3H8yvpWiarlNPbtWp1oRLicrIuMofibNHg/640?wx_fmt=png&from=appmsg#imgIndex=12)

▲图5 | 域随机化技术：在仿真中刻意注入各种随机噪声，强迫策略网络学会适应各种恶劣的物理条件©【深蓝具身智能】编译



形式化地，设仿真环境可调参数向量为 ，参数服从先验分布 ；

参数维度覆盖机器人刚体质量、地面摩擦系数、电机响应延迟、传感器高斯噪声幅值等。

域随机化的优化目标为在参数分布  下最大化期望累积奖励：





实际训练流程中，每个 episode 初始化阶段会对  的各维度独立随机采样，等价于让策略在数千组参数互不相同的仿真 “平行宇宙” 中迭代试错，例如：



- 把机器人的质量随机增加或减少 20%；
- 把地面的摩擦系数在“冰面”和“砂纸”之间随机切换；
- 给传感器的读数加上各种随机噪点；
- 甚至模拟电机时不时的延迟或轻微卡顿。

在数千组极端恶劣的平行仿真环境中完成训练的神经网络，部署至实体机器人后，现实场景中常见的微小扰动与误差，大多不会对其运行造成明显影响。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX5ne3MfNYQBbic4xIYsEJDKpCRqQXk6gllicSqc7QiabhaIEuCXA1I4xsg/640?wx_fmt=png&from=appmsg#imgIndex=13)

## GPU 并行仿真

你可能会问，要经历这么多试错，还要在几千个平行宇宙里训练，这得花多少时间？

如果放在五年前，可能需要几个星期。但现在，只需要几个小时。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGfgkKTW8dTaQWOUqqVNMqGWQyQZLn0xomvJKG78eSpAycX9yicjcSjUlhsB3Se0d83kfQ3mUyCJey7HvZz8iaPwDxs2d1DibJic3R4/640?wx_fmt=jpeg&from=appmsg#imgIndex=14)

▲图6 | 基于 GPU 的大规模并行仿真：可以在单一显卡上同时模拟数千个机器人，将收集数据的速度提升了成百上千倍©【深蓝具身智能】编译

以NVIDIA Isaac Gym为例，整个物理世界和神经网络都被搬到了 GPU 上。在一张 RTX 4090 显卡上，可以同时模拟 4000 多个机器人。

它们在不同的地形上同时奔跑、摔倒、爬起，每秒钟能产生近 10 万帧的训练数据。这种算力上的大规模并行，让机器人强化学习的迭代速度产生了质的飞跃。

👇用强化学习控制四足机器人 · 工程实践课👇

以动力学建模为地基、强化学习部署为终点。依托MATRiX可微仿真平台，完整拆解URDF解析、浮动基座动力学、系统辨识、PPO策略训练、域随机化、摩擦前馈补偿、实机部署全流程。深蓝学院《四足机器人：从动力学建模到强化学习》课程，由IROS四足挑战赛冠军，英国纽卡斯尔大学正教授潘为主讲：![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfgq0xV1gBWg7xgw03HcmeMEwz4Iy9btYAR0jHce7YGPjWNOhjbob5jbGzP6libReJJUKOQmyxOop0w9kkll7cbdTnCb1e6beYY/640?wx_fmt=png&from=appmsg#imgIndex=15)

▲部分课程内容展示 | 点击图片了解

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nticBjjcwNOxbPYPicJibngQA9IMr5vbvK9PFfG2VVkfzcxxWD90X2DoEgNAiaYMnO9F2GwHLOVs2U9OQ/640?wx_fmt=png&from=appmsg#imgIndex=16)

## 一个值得认真对待的范式转变

## 回顾本文梳理的这套技术体系，有一个值得认真思考的问题：为什么这套方法在短短几年内就能把四足机器人的运动能力推进得如此之快？

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcjVCFoXAJmA8fMwXornLue6lCU1PWbCXcPgvVtCojEicRNpGUcbIvZiaKYWQ1uo2ibCVMhuuG1Ty6aQck0lSkuWhOTl8INOPn6lg/640?wx_fmt=png&from=appmsg#imgIndex=17)

▲图7 | 经过强化学习训练的四足机器人，已经能够自主适应雪地、碎石、溪流等复杂的野外非结构化地形©【深蓝具身智能】编译

答案并不只是"算力更强了"或"算法更好了"这么简单。

更根本的原因在于，研究人员找到了一种让机器人自己发现物理规律的方式，而不是试图把人类对物理世界的理解全部编码进规则里。

当奖励函数只告诉机器人"走快一点、别费电、别摔倒"，神经网络在数以亿计的仿真步骤中，会自发地涌现出符合力学规律的步态——这种步态，和动物在漫长进化中形成的运动模式高度吻合。

当然，这套范式也有它清晰可见的局限性。目前的强化学习策略，大多依赖于精心设计的奖励函数和大量的仿真数据。

换句话说，今天的机器人强化学习，仍然是一种高度任务专用的技术。每一项新技能，背后都对应着一套新的奖励工程和仿真环境搭建。

这也正是当前研究的核心矛盾所在：如何让机器人从"被精心设计的任务里学会一件事"，走向"在开放世界里持续学习多件事"？

这个问题，目前还没有令人满意的答案。一些研究者在尝试引入语言模型来自动生成奖励函数，另一些人在探索跨任务的通用表征，还有一些人在重新审视模仿学习与强化学习结合的路径。

无论哪条路最终走通，有一点是确定的：它的进展速度，已经开始让那些曾经认为"通用机器人还需要几十年"的判断显得保守。

编辑｜阿豹

审编｜具身君

![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=png#imgIndex=18)

**——栏目说明与约稿——**

深蓝具身智能《具身智能基础》：它听起来有些笨拙，不如“大模型”、“世界模型”那样响亮。所以，我们不追求每一篇都让你“大呼过瘾”，但希望每一篇都能为你添上一块实实在在的砖瓦。很多内容也许你学过，也许忘了，也许从来没有机会系统梳理——没关系，我们一起重温。

把这些构成具身智能骨骼的东西，一块一块重新捡起来，用今天的眼光再理解一遍。欢迎更多读者分享你的观点与洞察，欢迎关注[💙](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)[《具身智能基础》专栏](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)

投稿与分享，请后台私信联系我们，发送「**基础知识」。**

 *****同系列专栏**
[![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGehZibr7tZF2zldbePrhVEqzN7MibldHydGKe6nGybQEX1BRAILTtBAjAjRnXgTvkfibaHsrYOzt70Uiaiclh9cuFVkcIMdff3SoyXs/640?wx_fmt=png&from=appmsg#imgIndex=19)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=20)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**

**投稿｜寻求合作｜研究工作推荐：私信点击【商务合作】**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=21)

点击❤收藏并推荐本文***
