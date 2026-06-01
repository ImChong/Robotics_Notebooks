---
title: 机器人下一代数据入口，可能就是Ego：9 篇论文讲透第一视角技术路线
author: 具身智能研究室
date: "2026-06-01 08:56:00"
source: "https://mp.weixin.qq.com/s/4JQ1xa-cJ7J1ep_e4txNnA"
---

# 机器人下一代数据入口，可能就是Ego：9 篇论文讲透第一视角技术路线

作者微信 yzz010329 · 商务微信 jszn576

最近我一直在看和 **Ego / Egocentric** 有关的论文。看多了以后，我越来越觉得，Ego 这个词在具身智能里不只是“第一视角视频”这么简单。

如果只是把摄像头戴在头上，那它当然只是一个视角。但真正有价值的地方在于：**Ego 同时记录了人的视线、手、身体、任务过程、遮挡、接触和临场决策。**

从外部视角看，一个人只是把杯子拿起来；从 Ego 视角看，你能看到他先看哪里、手怎么靠近、什么时候调整姿态、物体什么时候被遮挡、失败前有没有犹豫。对机器人来说，这些细节很可能比“最终动作结果”更重要。

|  |
| --- |
| **我现在的判断是：Ego 会成为具身智能里最重要的数据入口之一，但它不会天然变成机器人数据。** |

它中间还要经过很多处理：人体轨迹重建、手部追踪、视角对齐、动作重定向、物理过滤、世界模型推演、任务意图抽取。也就是说，Ego 的价值不在“视频很多”，而在于它给了机器人一个更接近人类真实执行过程的观察窗口。

这篇我选了 9 篇论文，按四个问题来读：

|  |
| --- |
| 阅读路线  01**第一，Ego 数据到底怎么大规模采？**  02**第二，人类第一视角怎么变成机器人可用的策略数据？**  03**第三，为什么世界模型也开始区分 ego 和 world？**  04**第四，只看第一视角够不够？** |

|  |
| --- |
| 01Ego 首先解决的是数据采集问题 |

机器人数据最大的问题，是贵。

真机遥操作贵，专家标注贵，多场景部署更贵。尤其是移动操作、家庭服务、零售、厨房这些场景，机器人要真正泛化，需要大量自然、连续、带接触的真实任务数据。

这时候 Ego 的吸引力就出来了：**让人类自己成为数据采集者。**

它不要求每个场景都部署机器人，也不要求每条数据都来自昂贵的遥操作系统。只要人戴着设备完成真实任务，就能留下接近机器人未来要面对的执行过程。

论文 01AoE：让普通人随时随地采集第一视角数据

|  |
| --- |
| 📄 论文标题：AoE: Always-on Egocentric Human Video Collection for Embodied AI  🏛️ 机构：Ant Digital Technologies、CAS、Zhejiang University、Peking University、BAAI 等  🔗 项目/数据：论文中介绍 AoE 采集系统  📅 时间：2026 |

![AoE 采集系统](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVibzicZhVdCm05InE02kSUYU7liaIrAuyRQhFdlOGPLWQvM2SLtLW7HMicMWTwkiaO93ReX6ccFv91QD0s8Fn2b0Jv88v4WtL9icZac/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

AoE 这篇最值得看的，不是某个模型结构，而是它把 **Ego 数据采集** 做成了一套系统。

它用的是更低门槛的方式：颈挂式手机支架、移动端应用、云端自动标注和过滤。这个设计背后的判断很直接：如果具身模型要继续 scaling，不能只依赖实验室里的机器人采集。

**人类每天都在真实世界里完成大量操作任务。** 做饭、整理、搬放、购物、清洁，这些动作本身就包含丰富的视觉、接触和任务顺序。Ego 数据的价值，是把这些原本没有被机器人系统记录下来的经验，变成可以被整理、筛选和训练的数据源。

这里也有风险。手机采集的数据肯定不如专业设备干净，视角会抖，手会挡住物体，动作也不一定严格标准。但这类工作真正打开的，是一个规模问题：**如果人人都能低成本采集第一视角任务数据，具身数据的增长方式会发生变化。**

论文 02EgoLive：把第一视角数据往真实任务场景里推

|  |
| --- |
| 📄 论文标题：EgoLive: A Large-Scale Egocentric Dataset from Real-World Human Tasks  🏛️ 机构：Joy Future Academy, JD  🔗 数据平台：https://robotdata-market.jdcloud.com/console/market  📅 时间：2026 |

![EgoLive 数据集](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEU8APGCh6VsHuEXLiaIIcciaOn71EBwDvcUtLcAzdTIOxgtM6oJ2VOggg05GdTNicgdCXo5xyicGibhsZ3yiaauUOWkpRfKl8Wllu3hA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

EgoLive 更像是在回答另一个问题：第一视角数据不能只停留在“日常视频”，它要尽量靠近机器人未来要做的任务。

论文强调的是大规模、真实场景、任务导向和多模态标注。它覆盖家政、零售等真实工作场景，目标很明确：给机器人学习提供更接近部署环境的数据。

我觉得这类数据集有一个重要意义：它把“人类视频”从泛泛的视频资源，往 **机器人任务数据** 方向推了一步。机器人不只是需要看到世界，还要知道一个任务通常怎么开始、怎么推进、哪里容易失败、人在执行时会怎样调整。

|  |
| --- |
| **Ego 数据如果没有任务结构，就很难直接服务机器人；EgoLive 这类工作真正补的，是任务过程本身。** |

|  |
| --- |
| 02Ego 真正难的是从人到机器人 |

有了第一视角视频，不代表机器人就能学。

人的手和机器人的夹爪不一样，人的身体和移动底盘不一样，人的视角和机器人相机也不一样。更麻烦的是，人类第一视角里很多信息是隐含的：视线扫过去了，手已经预备好了，身体重心也调整了，但视频里未必能直接看清。

所以第二组论文，核心不是“采了多少视频”，而是 **怎么把人的执行过程转成机器人能用的策略数据。**

论文 03EgoMimic：把人类第一视角当成可训练的模仿数据

|  |
| --- |
| 📄 论文标题：EgoMimic: Scaling Imitation Learning via Egocentric Video  🏛️ 机构：Georgia Institute of Technology、Stanford University  🔗 项目链接：https://egomimic.github.io/  📅 时间：2024 |

![EgoMimic 总览](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWWDCeN0aeLvdico9q7cQmTRHI86iadhJ7KNgyTQEZDuwqKHbQzQAFQXs9T3QrKGgLQpuKZFavmNm5U6CA22c1c7Z2XXxYgxO5Ps/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

EgoMimic 的判断很大胆：人类戴着 Project Aria 眼镜采到的第一视角视频，加上 3D 手部轨迹，可以作为 imitation learning 的数据来源。

这件事的关键，是它没有只把人类视频当成高层意图。它更进一步，把人类数据和机器人遥操作数据一起训练，让策略同时吸收两种来源。

我比较看重这个点：**EgoMimic 把第一视角数据从“看人怎么做”推进到“让机器人跟着学”。** 当然，中间要处理人的手和机器人的运动差异、外观差异、数据分布差异，这些都不是小问题。但它证明了一件事：第一视角数据不一定只能做视觉理解，它可以进入策略学习。

论文里还有一个结果很有意思：增加人类手部数据，对任务泛化的帮助非常明显。这个信号很重要，因为它说明机器人数据不足时，人类 Ego 数据可能是一个很强的补充源。

论文 04EMMA：移动操作也可以借人类第一视角数据扩展

|  |
| --- |
| 📄 论文标题：EMMA: Scaling Mobile Manipulation via Egocentric Human Data  🏛️ 机构：Georgia Institute of Technology  🔗 项目链接：https://ego-moma.github.io  📅 时间：2025 |

![EMMA 总览](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXogklGsCorEt26fliagJ2aYZ9I9PzthzUmaSZmxJw4xRjzH61U7XXZD046MfAwg6iaN0ZoickiaCBczTTKXFj1tKEk87rGPLKQONQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

EgoMimic 更偏桌面和双臂操作，EMMA 则把问题推到移动操作。

移动操作最贵的地方在于：机器人不只是动手，还要移动身体。让机器人完成移动操作遥操作，成本比固定机械臂高很多。EMMA 的思路是用 **人类移动操作数据 + 静态机器人数据** 共同训练，绕开大规模移动机器人遥操作。

这篇让我意识到，Ego 的重要性不是局限在手部操作。只要任务涉及“人怎么走过去、怎么靠近物体、怎么把身体对准操作对象”，第一视角都会比外部视频更贴近执行过程。

**未来移动操作最缺的，可能不是某一个抓取动作，而是人如何在空间里组织身体和任务。** 这恰好是 Ego 数据最容易记录下来的部分。

论文 05Gaze2Act：视线可能是比语言更直接的意图信号

|  |
| --- |
| 📄 论文标题：Gaze2Act: Gaze-Conditioned Vision-Language-Action Policies for Interactive Robot Manipulation  🏛️ 机构：MARS Lab, Nanyang Technological University  🔗 项目链接：https://zuo-kuangji.github.io/Gaze2Act/  📅 时间：2026 |

![Gaze2Act 框架](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEUXZ1SsRmRibyEbnyICYNS2AC9UqKicZyFLCNgRhHdKyXfXdtRehlkl4thLIWVQQvNLwMHHVic7nhrQbJicpA5jCnncpoDlvJMaIOg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

Gaze2Act 很适合放在这篇文章里，因为它把 Ego 里最容易被忽略的信息拿了出来：视线。

很多时候，人不一定能把意图说清楚。桌上有几个相似杯子，语言很难精确描述“拿左前方那个杯子的边缘”；但人看向哪里，往往已经透露了目标和动作区域。

Gaze2Act 做的事情，是把人的第一视角 gaze 映射到机器人视角里，再作为 VLA 策略的条件输入。这样一来，人和机器人之间的交互就不只靠语言，而多了一条更自然的意图通道。

|  |
| --- |
| **我觉得这类工作会越来越重要：语言负责说任务，Ego 里的 gaze 负责补细节。** |

对于机器人来说，真正难的常常不是“听懂拿杯子”，而是知道到底拿哪个、从哪里接近、什么时候目标变了。视线就是一种低负担、高密度的信号。

|  |
| --- |
| 03Ego 进入世界模型以后，问题开始变深 |

前面几篇更多在讲数据和策略。再往后，Ego 开始进入世界模型和长时程任务。

这时候问题会变得更复杂：机器人不能只理解自己看到的下一帧，还要判断 **世界本身在怎么变，自己的身体又在怎么影响世界。**

如果这两件事混在一起，长时程任务很容易崩。比如机器人移动时，画面变化有一部分来自相机运动，有一部分来自物体被推动，还有一部分来自环境本身的稳定结构。世界模型如果分不清这些来源，预测就会越来越乱。

论文 06Ego-Vision World Model：第一视角深度图服务接触规划

|  |
| --- |
| 📄 论文标题：Ego-Vision World Model for Humanoid Contact Planning  🏛️ 机构：UC Berkeley、University of Michigan、CUHK  🔗 项目链接：https://ego-vcp.github.io/  📅 时间：2025 |

![Ego-Vision World Model](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEX9kYlVupicjDULctkFkdibajaDxxzf8ajGuNupBHvibD72BQzK4mVIzlhmDTZ1CAShqPEZcKncC3KV2nYaNlb8k1Qw65OW1DyNiao/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

这篇论文讨论的是人形机器人接触规划。它不是让机器人避开所有接触，而是让机器人学会利用接触：扶墙保持平衡、挡住飞来的物体、低头穿过障碍。

这里 Ego 的作用很明确：机器人用 proprioception 和第一视角深度图做实时规划。它需要知道自己身体周围有什么、哪些地方可以接触、接触之后未来会怎样。

我喜欢这篇的地方，是它把 Ego 从“看见世界”推进到“为身体决策提供局部未来”。人形机器人很多能力都发生在身体附近：墙在哪里，障碍离头多远，手臂能不能挡住物体，身体会不会失衡。

**外部全局视角当然有用，但人形机器人真执行时，最后还是要靠自己的身体视角做局部判断。**

论文 07World-Ego Modeling：长时程任务里，世界和自我要分开建模

|  |
| --- |
| 📄 论文标题：World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks  🏛️ 机构：CAS、UCAS、Zhongguancun Academy、Shanghai Jiao Tong University、Peking University  🔗 项目链接：https://zgca-hmi-lab.github.io/WEM  📅 时间：2026 |

![World-Ego Modeling](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVD84iahoO09wSkfiaTlol5o4DmOicPdaKIyjEnb2xHIyM0lnLKrAPxib70SjpWOvmoaCzWh8UAHKtIF3aOWVyuPEBsRckJKhEcMpk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

World-Ego Modeling 这篇的名字就很直接：把 world 和 ego 拆开。

它的核心问题是长时程任务。导航和操作交替发生时，场景里有一部分东西是稳定的，比如房间布局、物体位置、上下文；另一部分是和机器人自身动作强相关的，比如相机视角、手的运动、被操作物体的变化。

如果世界模型把这些都塞进同一个流里，很容易出现长期漂移。WEM 的思路是把 **世界演化** 和 **自我演化** 拆开建模，让长时程 rollout 更稳定。

|  |
| --- |
| **这也是 Ego 重要性的另一面：机器人必须知道哪些变化来自世界，哪些变化来自自己。** |

这句话听起来简单，但对具身智能很关键。只会看视频的模型，可能会把相机晃动、身体运动、物体变化混在一起；真正能执行任务的模型，需要分清“我动了”和“世界变了”。

|  |
| --- |
| 04只看 Ego 也不够 |

写到这里，容易把 Ego 说得太万能。

但第一视角有天然盲区：看不到自己的全身，容易被手遮挡，空间记忆不稳定，外部关系也常常不完整。一个人戴着眼镜操作时，很多信息其实来自大脑的长期记忆和第三视角经验，并不全在视频里。

所以最后两篇，我更愿意把它们看成提醒：**Ego 很重要，但 Ego 需要和 exo、3D memory、环境结构结合。**

论文 08EgoExoMem：第一视角记忆需要外部视角补足

|  |
| --- |
| 📄 论文标题：EgoExoMem: Cross-View Memory Reasoning over Synchronized Egocentric and Exocentric Videos  🏛️ 机构：KIT、ETH Zurich、University of Oxford、Hunan University、INSAIT  🔗 代码/数据：https://github.com/RuipingL/EgoExoMem  📅 时间：2026 |

![EgoExoMem](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEW4uwm4EtlLibGECh8YvbC74lqLy1NQSUfBEVicvVyKQA2Pad4SpcWG2OicyCJyahCjSL5up0Xo1O2Sy91kicZFnKOqmnLLI7pdDec/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

EgoExoMem 这篇非常适合给 Ego 降温。

它研究的是同步的 ego-exo 视频记忆推理。很多问题单靠第一视角答不出来，单靠外部视角也答不出来。第一视角知道操作者看到了什么，外部视角知道人与物体的整体关系。

这个结论对机器人也很有启发。只看机器人自己的相机，很多空间关系会丢；只看外部摄像头，又容易丢掉机器人的局部意图和接触细节。

所以我觉得未来的数据系统，很可能不是单纯押 Ego 或 Exo，而是走向 **Ego + Exo + 3D 环境记忆**。Ego 负责临场执行，Exo 负责补全结构，环境记忆负责把短片段连接成长时程理解。

论文 09E³C：生成第一视角视频，也要有 3D 环境记忆

|  |
| --- |
| 📄 论文标题：E³C: Video Generation with 3D Environmental Memory and Ego-Exo Human Pose Control  🏛️ 机构：Meta Reality Labs、University of Toronto  🔗 项目链接：https://e3c-videogen.github.io  📅 时间：2026 |

![E3C 视频生成](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWVxHJiaPPcpsWT1ZLP5XMC1QKpsjcvceUSrp9lDciaLtotmfOvwQU3zcsE9UCrgERAmHyvkyADVZ7Hex6K3ibYwqhDp26EsfaOQM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

E³C 讨论的是可控第一视角视频生成。它的关键词有三个：3D environmental memory、ego human control、exo human control。

这篇让我比较在意的是，第一视角视频生成不能只追求画面像。Ego 视频里相机和身体绑定，视角变化快，遮挡多，人的动作还经常只露出一部分。如果没有 3D 环境记忆，生成视频很容易在空间上漂。

放到机器人里看，这件事更明显。机器人要靠世界模型想象未来，生成出来的未来不能只是好看，还要能保持场景结构、身体动作和物体变化的一致。

**第一视角生成视频的下一步，不是更像短视频，而是更像可用于机器人推演的世界片段。**

|  |
| --- |
| 05写在最后：Ego 重要，但它不是捷径 |

把这 9 篇放在一起，我对 Ego 的理解比之前更清楚了一点。

Ego 重要，首先是因为它能把人类真实任务过程记录下来。它看到的不只是结果，还包括执行前的注意力、执行中的遮挡、失败前的调整，以及身体和物体之间那些很细的关系。

Ego 重要，也因为它更接近机器人未来自己的感知方式。机器人最终在真实世界里工作时，也会从自己的相机、深度、触觉和本体状态出发做判断。第一视角数据天然贴近这种“从身体内部看世界”的方式。

但 Ego 不是捷径。**第一视角视频不会自动变成机器人数据。** 中间还需要手部追踪、身体重建、视角对齐、动作重定向、物理一致性检查、世界模型推演和策略学习。

所以我现在更愿意这样理解 Ego：

|  |
| --- |
| **Ego 不是具身智能的答案，但它很可能是具身智能规模化数据的入口。** |

如果未来机器人要从人类经验里学习，不能只看外部视频里“发生了什么”，还要理解执行者在任务过程中“看到了什么、准备做什么、怎么调整身体、为什么失败”。这正是 Ego 最有价值的地方。

![作者微信二维码](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWlBT7M39GWaakK7FqVEQ1KlRa4d7HuBnwNIu3F39vfAraGS9oiceLLpg5EiaZEegYBn2kibK9MFOiaRtic3kol9ibKwazuH60iaMx5D0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)
