---
title: 目前最接近具身 ICL 如何实现的必读论文解读
author: 每日智能
date: "2026-08-31 00:58:33"
source: "https://mp.weixin.qq.com/s/vIUalf3vZI3AV-HWSVruew"
---

# 目前最接近具身 ICL 如何实现的必读论文解读

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1TutaicAEA2SZ7OhMWF3eylCB8McVCIC8YzndiaicMuG6u2LeXt0kUUpP0ibw7445mfibEMWMppvFpw0qFG0h7tXFCiaRIGca3w3VEicdSib8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

由于GEN1.5和S1讲的具备GPT-3能力的 [#EICL](javascript:;) (Embodied In-Context Learning) 并没有直接的论文，截止到2026年8月底，目前最接近具身 ICL 如何实现的论文包含以下四篇：**WAM-TTT、RoboTTT、StellaVLA、Zero-WAM**。这四篇论文把「演示当提示」拆成了四条互不相通的路线，而这是它们的第一次纵向剖析和横向对比。

> 这篇文章讲四篇论文的机制、数字、消融、以及各自没说出口的那部分，全部收进同一套坐标系里；文末给出四列大表、漂移轴地图和一张列着七个空缺的「未走之路」清单。
>
> 全文总结 9 张图 和 1 张四列大表 · 数字均为论文自报 · 跨篇不可直接比较 · 阅读约需 25 分钟


![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1TutYYnK4eukqcibrl7INZovmS8BAMGcXIOOhmUW6jhG817MribKooXiczCVevGzIzicPjW3LK4iblZediadoApXoZqq1J6Q4xgOBKRrddM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)图 1 同一个输入，四条通路。四篇论文都在回答「一条人类演示怎么变成机器人动作」，但演示的落点完全不同：前两篇把它写进权重，后两篇让它停留在输入里。

---

## 引子

产业侧先把话说满了。

Generalist AI 在 GEN-1.5 的博客里写 *Embodied Foundation Models are One-Shot Learners*：一条 3 到 12 秒的演示当作「物理提示」，10 个任务上一次性 in-context 提示平均 **59%** 成功率（±10%），再用 5 分钟数据跑 10 步梯度下降升到 **83%**。而且博客强调，这些能力是从 8 个月连续预训练里**涌现**的，无架构改动、无 meta-learning 循环、无辅助目标。

Skild AI 的 S1 说得更远：一条视频演示当 prompt，能做**预训练中从未见过的 10 分钟长时程任务**，成功率 **66%** ，同数据规模下语言提示只有 **9%** ，已见任务约 **96%** ；并称单条 in-context 演示约等于 380 条后训练示例，而采集那 380 条长时程演示要花 50 到 100 小时遥操。

两家都没有同行评审论文，没有开源权重，没有 API。证据形式是 demo 视频加内部曲线。

过去两个月，四篇论文把同一件事做成了可以核对的实验。它们的问题其实只有一句：**一条演示，怎么才能真的被策略用上。** 四篇给出了四个互不相同的答案。更值得玩味的是，它们彼此知道对方的存在（Zero-WAM 甚至专门写了一段与快权重派划清界限），却**没有任何两篇做过直接对比**。

> **读之前先约定。** 四篇的绝对数字**不可跨篇比较**。泛化的定义（未见场景 / 未见构型 / 未见扰动 / 未见任务）、指标口径（部分给分的 progress、rubric 完成分、二元成功率）、任务难度轴、本体数量、试验次数，全都不同。本文所有横向表格都是**定性定位**，不是性能排序。

---

## 一、坐标系：演示到底被放在了哪里

把四篇的机制剥到最里面，分歧只剩两个维度。

第一个是**演示最终改变了什么**。要么改权重：把演示压进一小组参数，模型自己变了；要么改输入：模型一个字节都不变，只是看到的东西变了。第二个是**适应发生在什么时候**：部署前做一次，还是 rollout 里每一步都做，还是根本不做。

这两个维度一交叉，四篇各占一格，没有重叠。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1TutbicJLY8ft8AOJaJk1LovkyPxM1dj1DzK41j3hPOrSefxhCE8uegXdaLaTJedDIUZWcrtm3r9u5n7ScH93hDSJWubtOCicW8CF5o/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)图 2 四篇各占一格。左半区是快权重派，测试时仍有梯度（但只对一小组快权重，主干全冻结）；右半区是纯上下文派，测试时一个参数都不动。GEN-1.5 与 Skild S1 在博客里主张的路线落在右下角，与 Zero-WAM 最接近。

---

## 二、WAM-TTT：把演示写进记忆，而且只许写在感知侧

> 论文信息：arXiv:2607.06988　北京大学 · Galbot 银河通用 · 中科院自动化所 · 清华　2026-07
> **关键词：快权重 · 感知侧 · 离线技能吸收**

> 与其把人类视频当轨迹去模仿、或者当上下文 token 塞进去，不如通过自监督视频预测，把它吸收进一个冻结世界-动作模型内部的轻量快权重记忆里。

![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1TutbL0hTDHOic5VariaD2OtiaYxA9lQKJLoyzhBLdoq1aeYzD584QCD3jsFxvyBuBMT6UkwpHibM3QVAueiaFNpaFUYyqtuAbLgfBonGI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)WAM-TTT 流程

它的形式化很干脆：给定一小批**无动作标注**的人类视频，测试时只优化视频专家上的 TTT 快权重（默认 N=1 步 inner SGD），主干、慢投影、快权重初始化、动作专家全部冻结；随后拿适应完的固定快权重去跑机器人。输入只有第一人称 GoPro 的 RGB 帧，没有手部姿态、没有接触、没有 3D 场景线索、没有重定向轨迹。

### 最扎实的一处贡献在附录里

内层损失除了标准的人类视频预测，还多了一项**键值记忆重建损失**（KVM），衡量当前快权重能不能从人类 Key 重建人类 Value。论文在附录 A 给了线性特例的闭式解 ᵀᵀ⁻¹，在各向同性假设下它退化成 Hebbian 外积记忆，读出形式恰好是 ᵢᵀᵢ，也就是一个**没有 softmax 的线性注意力**。

这意味着 KVM 损失不是随手加的正则项，而是「用快权重去实现对人类 Key/Value 的交叉注意力」的变分定义。它把「演示当上下文」和「演示当记忆」这两种看起来对立的做法缝在了一起。四篇里概念上最漂亮的一步就是这个。

### 架构上的保守主义

TTT 残差**只加在视频专家**，动作专家一行不改。这个选择解释了它为什么抗域漂移而 co-training 会崩：人类侧的域漂移只能改写「人到机器人」那份记忆，改写不了策略本身。这条经验后面还会再出现一次。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1TutZU79pibElRWEwbMpN1uDjpTMDZxJPYMku5hR915S3hia3yEicUuJu13CPUuL0aSP0y1GKabSnVRWlWYicIAo0wUgsQp9SgeqyDgUs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)图 3　左：TTT 残差只挂在视频专家上，动作流完全不动，这是它抗遗忘的结构性原因。右：New 设定（未见家庭 / 厨房 / 办公室）下 9 任务平均 progress。注意 progress 是部分给分指标，不是二元完成率。![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1TutbkDWlExnfklrVNIvuxYVKfibjVu5XoCffEWgwpl5xX42cXrAMss8efMuoTxbrVWn5VItF7stAw71P6eZLknmALWHfGXicFJl0KE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

### 三条消融比主结果更有信息量

- **去掉 meta-training 几乎归零**：Table Bussing 从 100.0 掉到 9.0，Swap Place 从 88.9 掉到 0.0。预先学好那个「人到机器人」的记忆接口是不可省的。
- **通用 LoRA 换不掉 TTT 快权重**：30.0 / 0.0。起作用的不是「多了一组可调参数」，而是那个记忆结构本身。
- **给人类侧加伪动作是净负面**：加 MANO 重定向伪动作和前向动力学损失，四任务均值从 72.3 掉到 28.9。在当前单视角手部追踪的成熟度下，保持人类视频「无动作」、只用抗噪的视频预测损失，是对的。

还有一条对产业最实用：配对人类数据可以 **1:1 替代机器人数据**。等预算 200 条 episode 下，(机器人 100, 人类 100) 得 74.1，(机器人 200, 人类 0) 得 73.7，统计上不可区分，遥操成本直接减半。但 (10, 190) 只有 51.4，说明人类数据是补充而不是替代，机器人 grounding 仍然要有。

> **该打的折扣:**
> progress 是部分给分，权重偏向少数关键里程碑（Pour Water 的「倒水成功」一项就占 0.60），真实的全完成率会显著低于 46.2。「零梯度」的说法也要小心：它是**零主干梯度**，测试时仍然对快权重做 SGD。另外 WAM-ICL 这个靶子本身就不是为长上下文 ICL 预训练的主干，被打到 7.1 有一部分要算在主干不对口上。最后，没有代码、没有项目主页、没有数据集，外部无法独立复现。

---

## 三、RoboTTT：把上下文长度变成一条新的 scaling 轴

> 论文信息：arXiv:2607.15275　NVIDIA GEAR · Stanford · UT Austin　2026-07
> **关键词：快权重 · 动作侧 · 流式工作记忆**

> 它的主命题其实不是具身 ICL，而是「上下文长度是机器人基础模型的新 scaling 轴」。one-shot ICL 只是长上下文足够长之后涌现出来的副产品之一。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1Tuta7UFQ3fCR3FhxptAX3bWUXT6rUcNBakH0UBjQ52HLTtfWBPVicHl7bbcfY8nbsFrrLbSKRXeOUL2YibqE7yH0sBA6MIUdPbRtds/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)RoboTTT 流程

它把视觉运动上下文从主流 VLA 的单帧几帧拉到 **8K 时间步**（30 Hz 下四五分钟），比 SOTA 高三个数量级，而且推理延迟不随上下文增长。另外三项副产品是在线自我纠错、抗外部扰动、长时程多阶段任务。

### 同叫 TTT，和 WAM-TTT 不是一回事

TTT 层加在 **DiT 动作头**每层注意力之后。分工可以用一句话概括：**注意力在单个时间步内做，TTT 层跨时间步做。** 快权重在 rollout 的每一个时间步被梯度下降更新并向前传播，本质是一个「递归状态即快权重」的 RNN。跑到第 9000 步时，快权重已经被更新了 9000 次。

放在一起看：**WAM-TTT 的快权重是技能包，RoboTTT 的快权重是工作记忆。** 一个在部署前装好就冻结，一个边走边写。

### 让它训得起来的三件事

- **tanh 软门控**（α 初值 0.001）：训练初期 TTT 分支贡献接近零，逐步放开，保住预训练能力。WAM-TTT 用的是硬性架构隔离，这里是学出来的软门控。
- **sequence action forcing**：每个动作 chunk 独立采样 flow-matching 噪声水平。不这样做训练就不稳定，消融显示去掉它动作精度直接崩到无法推进任务。
- **TBPTT**：梯度只在段内流动，快权重跨段携带。显存由段长而非总长决定，于是固定显存预算下训练上下文可以任意长。

### 最漂亮的是一个开关

只靠「掩掉某些时间步的 flow-matching 损失」，三种异构上下文就被统一进同一套训练：人类演示视频（只更新快权重，不算动作损失）、episode 内自己的 rollout 历史、DAgger 轨迹里自己的失败动作。

由此长出来的 **DAgger Distillation** 可能是整篇里最容易被别人直接拿走的一块：标准 DAgger 只在人类纠正上微调、丢掉机器人的次优动作；这里反过来，**失败作上下文，纠正作目标**。同一批 100 条轨迹，标准用法平均 +9%，这个用法 +33%。而且它在 GDN 上也有 +29%，说明这条不绑定 TTT，可以接到任何序列模型策略上。

![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1TutbOb3x2oQ4dSBZ5EUzeOGBFahTMn5IdYkxWn0LvvVEJp6YibOSXmcOrxC1EmicdZQzaWW4cjN2QTj1EjoSUg2sicgRGficUEfRRiaTM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)图 4　左：TTT 层跨时间步工作，快权重沿 rollout 递归更新；同一套机制靠损失掩码接入人类视频段。右：预训练上下文 1K 到 8K，平均完成分从 43.9 升到 71.5，曲线无饱和。层位置、门控、参数量全部对齐的 GDN 对照吃不到这条 scaling，说明并非任何定长状态都能利用长上下文。![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1TutbcmPccDiay5VxcZBDANy3DXEZ39z4SopoKTYtWRonRWX1NUISS0h5v59fICLu5G5zafy0ibzAT0G7libcKIs33SsNnsLJK61Rz8k/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

还有一条反直觉的数据值得记住：**朴素拼历史反而有害**。GR00T N1.7 Hist.（多给一帧历史）在 Pup Go Car 上只有 39.5%，低于无历史版本的 57%。论文归因于历史引入伪相关和推理时的时序 OOD。长上下文不是免费午餐，这个坑后面还会撞到两次。

> **该打的折扣：**
> **8K 指的是预训练上下文**，主结果的部署策略是在 1K 上下文下后训练出来的，不要读成「部署时用了 8K」。one-shot 那一格的人类演示视频，是**机器人保持静止、由机器人自己那 4 路固定相机拍摄的**：视角、场景、光照全部同源，只差「操作的是人手还是夹爪」这一个变量。而且「未见」指的是同一块电路板上 80 种元件构型里的未见组合，不是未见任务、未见场景、未见物体。基线三个全是自家主干的变体，没有跟任何专门为 ICL 设计的方法比过，也没跟全注意力长上下文比过（理由是解码延迟，这是工程理由不是精度理由）。
>
> 所以准确的说法是：**RoboTTT 证明了梯度式快权重优于线性递归状态，没有证明它优于全注意力上下文。**

---

## 四、StellaVLA：把演示翻译成「为什么」，而不是压进权重

> 论文信息：arXiv:2608.11671　StellarEdge AI（公司技术报告）　2026-08
> **关键词：上下文条件 · 结构化语言 · 零梯度**

> 把检索到的演示当原始观测和连续动作直接拼进上下文，只传达了专家做了什么，会鼓励表层模仿；应该离线把它转成推理增强的结构化上下文，传达为什么。

![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1TutaKwKKNIjSqQr7IoOqDTdS1VA5B6E0E7EEgRibTJeTOAAaiaNAj3FEaxibToJ4WHaucTOj5Pb9TMFDdz95cbgG1v6BJ3gF3shnZug/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)StellaVLA 流程

一条全自动离线流水线用 Qwen3-VL 把每条轨迹拆成 K 个语义段，每段生成两层理由：**语义理由**（子目标描述，比如「伸向蓝色马克杯的手柄」）和**运动学理由**（3D 工作空间位移加 2D 投影轨迹的文字化）。零人工标注。测试时检索一条这样的结构化演示当前缀，主干全程冻结，没有快权重，没有 inner loop，**这是四篇里唯一真正零梯度的**。

### 最精巧的一步：同一个算子用在两个尺度上

确定性的 verbaliser Φ 把任意连续动作跨度映射成 3D 位移和 2D 投影。在**整段**上应用得到进上下文的运动描述；在**动作 chunk** 这个短得多的尺度上复用同一个算子，得到训练监督目标。于是「别人的演示」和「策略自己的预测」被放进了同一套词汇里，天然可比。

### 它做了一个别人都没做的实验

同一个训练好的 checkpoint，评测时只换演示内容：**正确演示 98.8，无演示 62.4，错误演示 44.9**。

关键在于「错」比「无」更差。如果策略基本忽略上下文，缺失和错配应该给出相近的结果；多出来的那段退化说明它**主动**在用演示确定意图，哪怕这个上下文是误导性的。逐 suite 看更清楚：Goal suite 从 99.6 掉到 24.8（无）再到 0.0（错），而 Spatial 在无演示和错演示下几乎一样（72.2 / 72.6），因为当前观测本身已经告诉了它该在哪儿交互。

这个三向干预实验应该成为所有 ICL 类论文的标配自检，用来证伪「策略其实忽略了上下文」。四篇里只有这一篇做了。

![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1TutbWUGoj5wnDiazj0Ric4BXuialbMBzNib56iaVxxcJKVn9SAZGmG4hgUkGDmfaP8IGz6NKqAssicDw5HJ1myh5sMNuQvZ0jD4L65icKJo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)图 5　左：演示先被离线转译成结构化语言，再当作固定前缀，主干与动作头之外的语言专家在部署时整个拆掉。右：三向干预。错误演示比没有演示还差 17.5 个点，这是四篇里唯一直接证明「策略确实在使用上下文」的实验。![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1Tuta73D0jvFJ8bdQebb2Tia4KL9CSfjyV3cSl7ULEWU0ErUzqrW5eNOR8AI6pyIn1ljuMFxWbqX3fJAmQ8HcpVI1WG5XzcfMKjVv0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

### 可迁移的信息几乎全在语言里，不在像素里

模态消融同样干净：评测时 **Text-only 98.8 / 84.4** 几乎等于完整的 Image+Text 98.8 / 85.1，而 **Image-only 塌到 92.9 / 75.7**。训练时用 Image-only 分内略好（98.4 vs 97.3）但 OOD 差很多（78.7 vs 85.0），说明视觉演示会让策略走「外观对应」的捷径，分内有效、扰动下不可靠。这也是它敢做跨本体上下文的底气：外观信息在进入检索之前就被抽掉了。

LIBERO-Plus 上的增益分布最有解释力：相机视角 +23.5、传感器噪声 +19.7、机器人初始状态 +14.7、语言扰动 +8.3，**但物体布局只有 +0.1**。论文自己的解释是「演示中场景特定的空间关系不再匹配当前场景」。记住这一条，它在总览里会变成一个关键坐标。

> **它自己的消融拆了招牌的一半：**
> 论文两大贡献之一是「并行双训练把推理内化」，但 Table 6 显示 LIBERO-Plus 的 OOD 鲁棒性在 **λ=0 时最高（86.9）**，随 λ 增大单调下降到 81.9，选用的 λ=0.3 只有 85.1。也就是说：语言监督换来的是饱和基准上 +1.2 的分内提升，代价是 −1.8 的 OOD 鲁棒性。**真正在起作用的是上下文条件本身，不是那个辅助 loss。** 论文诚实地报告了这件事，但摘要和贡献列表仍然把它当卖点。如果你的目标是泛化而不是刷榜，先试 λ=0。

> **其他该打的折扣：**
> 仿真里指令集是封闭的，**检索退化成了精确任务匹配**（附录 A 明说），所以 98.8 vs 62.4 里有相当一部分在测「知不知道任务标签」。VLA-Arena 上的 0.63 vs 0.44 存在测试时信息不对称：基线数字取自 leaderboard 未重跑，且只有它在测试时额外拿到一条目标任务的演示，而那个基准上没有 matched control。长时程是它的崩塌区：VLA-Arena Long Horizon 的 L1 只有 0.02、L2 是 0.00，作者给的原因很到位，**固定前缀能指定程序，但无法在执行漂移后重新规划**。真机每格只有 10 次 rollout，作者自述单格有 10 到 15 点噪声。算力完全未披露，是四篇里唯一一篇。

---

## 五、Zero-WAM：把人类视频当任务规格，并且自己造出来

> 论文信息：arXiv:2608.26103　Robbyant · 香港科技大学（广州）· 香港科技大学　2026-08
> **关键词：上下文条件 · 原生人类视频 · 零梯度 · 跨任务**

> 零样本跨任务泛化本质是任务规格问题。而对操作任务来说最自然的规格不是语言而是人类视频：语言难以刻画空间约束、中间状态与时序结构，也无法提供「场景应当如何演化」的直接视觉证据。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1TutZibicDicDvI0un5oSl6Le5eyeyWET9owib3wVQib1Atz31dIIH4KAic5ZzKdWIZXica0PCMpSia2HUhoM8zYdnts6KWLFiccVdibvcZh18o/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)Zero-WAM 流程

四篇里只有它正面攻打**未见任务**。前三篇的泛化分别是未见场景、未见构型、未见扰动；它做的是 RoboTwin 2.0 上的**任务级切分**：43 个任务训练，7 个任务完全留出，测试时不给任何该任务的机器人数据、不更新任何参数。7 个未见任务平均成功率 **46.95%** ，对照的 LingBot-VA 是 17.45%、WAN-Action 是 10.98%，而且七个任务全部领先。评测规模也是四篇里最大的：3 个随机种子 × 每任务 100 次闭环 rollout。

### 它的核心不是模型，是数据流水线

主流做法是采人类视频，再想办法给它配上机器人动作。Zero-WAM 把方向**反过来**：从已经有可执行动作的机器人轨迹出发，用 VLM 分析任务、用图像编辑模型把首帧改写成人类操作场景、用视频生成模型合成人类操作视频、再用 VLM 打语义分和物理分做质检。产出 **74.2K 对人-机器人 ICL 配对，覆盖 8.6K 个任务、超过 45 种本体**。作为对比，RH20T 的 110K 样本只有 147 个任务，EgoScale 是 10.3K / 344。任务覆盖高出约 25 倍。

课程设计也很聪明：External 子集**刻意加大**视觉不对齐（强制改场景、背景、物体实例与摆放），逼模型学「不管长什么样，任务是什么」；In-house 子集反过来**刻意保持**对齐，学精确对应。

### IFP：全文最关键的一个机制

teacher-forcing 训练时有一条捷径：下一段机器人视频往往只靠外推最近的机器人历史就能预测出来，模型于是可以不读人类视频也把损失降下去，到了未见任务上继续依赖机器人历史，**恰好在最需要 ICL 信号的时候不用它**。IFP 是一个训练期专用的辅助目标，要求从当前机器人视频表示预测多个跨步的未来 chunk。

最精妙的一处决策是：IFP 模块**刻意不直接以人类视频为条件**。理由值得完整复述：如果它直接吃人类视频，这条辅助分支就会自己学出一个独立的「人类视频条件未来预测器」，而不必逼主视频 Transformer 去编码 in-context 任务信息；由于 IFP 在推理时被移除，部署的策略就什么也没得到。这段推理对任何「训练期挂辅助目标、推理期剥离」的工作都通用。

![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1Tutaym9q5ZBRJ5iapxUYS8UxqlicerR7rvtS3bD3fO2ovOpQkibRD4rX10A8AAz51ibjr152DXrAheDzxUWUwoUdKS9F2XAdAprMsdIk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)图 6　左：人类视频只进视频分支，动作分支的条件完全不变。右：把论文自己两张消融图的数字串起来。头条 +29.50 里约 22 分来自任务级重采样这个数据工程决策，只有约 7.5 分来自标题所主张的 in-context 人类视频机制；而去掉 IFP 之后，加人类视频（28.55）反而比不加（39.44）低 10.9 分。![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1Tutb2KoibIn7Leicr6deRDvgXZxxBMic8Dq07gLbynXiahJib26LEvutRt6ZKmfBibn8WNPjZwg1cbGKS3DdhP1Yg4au1wpNDeSf9icf97w/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)
> **最尖锐的一条：**
> 把两张消融图并排算：LingBot-VA 17.45 →（只做任务均衡采样的 text-only 变体）39.44 →（加人类视频但去掉 IFP）28.55 →（完整）46.95。**朴素地把人类视频塞进上下文反而损害跨任务性能，是 IFP 把它救了回来。** 这其实强力印证了论文自己的捷径学习论点，但也说明「人类视频作为任务规格有效」这个主张，比摘要读起来脆弱得多。
>
> 需要限定：这两个数字来自不同的消融图，论文未声明两个变体的训练配置完全匹配，也从未把它们并列过。方向性结论应当成立，严格的等价性需要作者确认。

> **其他该打的折扣：**
> 人类视频 **100% 是合成的**，而测试时给的人类视频到底是真人拍的还是同样生成的，论文没有明说，**合成到真实的 gap 完全未评估**。质检环节没有任何数字：通过率、人工抽检一致性、错误类型分布一律未报，「零人工标注成本」的另一面是「零人工质量把关」。HumanGen 也不增加任何新任务知识，人类视频是从机器人轨迹渲染出来的，它规模化的是**接口**而不是**知识**。唯一的外部基线 LingBot-VA 与本文作者高度重叠，实质上是同组前作；两个任务的成功判据被修改过（对所有方法一致应用并承诺公开），而这两个恰是它得分最高的项目之一二。

> 数据流水线依赖: Gemini 3.1 Pro、Nano Banana 2、Kling AI 3.0 等闭源商业模型，外部团队连「自己重造一份 HumanGen」这条路都被堵住大半。

---

## 六、总览之一：四列大表

同一套 12 个维度，四篇并排。这张表的用法不是拿来排名，而是拿来定位：看清楚每一篇在哪个格子上是强的，以及它们的强项为什么互不重叠。

| 维度 | WAM-TTT 2607.06988 | RoboTTT 2607.15275 | StellaVLA 2608.11671 | Zero-WAM 2608.26103 |
| --- | --- | --- | --- | --- |
| **ICL 的定义口径** | 定义成测试时训练问题；演示进快权重记忆，明确反对上下文条件 | 主命题是上下文长度作为 scaling 轴；ICL 是长上下文之后涌现的能力之一 | 标准 ICIL；创新不在适应机制，在演示的表示形式（做了什么 → 为什么） | 零样本跨任务泛化 = 任务规格问题；人类视频是操作任务的自然规格 |
| **上下文模态** | 真实 GoPro 第一人称人类视频，无姿态、无接触、无重定向 | 异构三源：人类视频、自身 rollout、DAgger 失败动作，由损失掩码统一接入 | 结构化文本（子目标 + 2D/3D 运动）加子目标关键帧；消融显示纯文本几乎等效 | 合成的人类操作视频（原生像素）；ICL 模式下把语言指令整个关掉 |
| **是否零梯度** | 零主干梯度；rollout 前 1 步 inner SGD，之后冻结 | 零主干梯度；快权重每一步更新并前传 | **完全零梯度** | **完全零梯度** |
| **主干** | LDA-1B 世界-动作模型（Qwen3-VL-4B + DiT-L 双专家）；TTT 只加视频专家 | GR00T N1.7（Eagle VLM + DiT，538M → 690M）；TTT 加动作头，tanh 门控 | Qwen3-VL-4B 全参微调 + MLP 动作专家 + 可剥离语言专家 | Wan-2.2-TI2V-5B 改造成因果视频-动作模型；MoT 双分支 |
| **上下文机制** | 快权重 MLP 记忆；**KVM 损失等价于无 softmax 线性注意力** | 快权重即递归状态；TBPTT 推到 8K 步；16 个 register token 跨时间搬运 | 前缀 KV 缓存（183 → 91 ms）；子目标帧降到 3 帧仍有 98.1 | 人类视频作视频分支前缀；RoPE 高度轴偏移 ΔH=32；动作分支不 attend 人类视频 |
| **核心训练目标** | 人类视频预测 + KVM 记忆重建；相位对齐配对人机帧 | 逐步 flow-matching 平均 + sequence action forcing + 损失掩码 | L1 动作损失 + λ·语言损失（λ=0.3），两专家严格并行 | 视频/动作双 flow matching + **IFP 反捷径目标**（K=4，s=2） |
| **数据规模** | 2286 对配对 episode，9 任务；主干 EI-30k（30k+ 小时） | 总量未披露；下游任务 8 / 6 / 5 小时 + 30 分钟扰动 + 100 条 DAgger | 仿真用公开基准；真机 125 条遥操（71,702 帧）+ 52 条 XR 人手与重定向 | **HumanGen 74.2K 对 / 8.6K 任务 / >45 本体**，全自动生成 |
| **本体** | **3 种真机**：Unitree G1、Galbot gripper、Galbot sharpa（58 维） | 1 种：YAM 双臂桌面，4× RealSense，30 Hz | 1 种：AgileX Piper 6-DOF；但上下文源有 3 种 | 1 种真机：双臂 Franka；训练数据覆盖 >45 种 |
| **泛化维度** | 环境视觉漂移为主：未见家庭 / 厨房 / 办公室、新物体、光照、桌高、跨末端执行器 | 时间长度为主：1 到 5 分钟、最多十阶段；80 种构型训 20 测 60；扰动恢复 | 扰动轴与任务规格：LIBERO-Plus 七轴、VLA-Arena L0 到 L2、跨本体上下文源 | **未见任务**（任务级切分）、未见物体容器、未见时序、未见插装构型 |
| **评测严格程度** | 9 任务 × 2 设定 × 25 trial；只报 progress 部分给分；rubric 权重未完整披露 | 3 任务 × 20 trial；**兼报二元成功率**；附录完整披露 rubric 权重 | LIBERO 每 suite 500 rollout；**独有三向干预 + matched control**；真机每格 10 | **3 种子 × 每任务 100 次 rollout**，报标准差；真机 30 次 / 族但无干预对照 |
| **开源程度** | 差：无代码、无项目页、无数据；仅主干开源。8× H800 | 中等偏下：有项目页与完整 rubric；无代码权重数据。16× GB200 | 最差：仅公司博客页；算力完全未披露 | 差：有项目页；唯一报 GPU 小时（15,360）；流水线依赖闭源商业 API |
| **一句话核心洞见** | 演示当快权重记忆比当上下文 token 更抗真实漂移：46.2 对 7.1，保持率 76% 对 15% | 上下文长度是新 scaling 轴，而且只有梯度式快权重吃得到：43.9 → 71.5，无饱和 | 演示抽象成结构化语言后作为任务规格起作用：错 44.9 < 无 62.4 < 对 98.8 | 合成配对能把 ICL 任务覆盖推到 8.6K；但捷径严重到没有 IFP 时人类视频是负收益 |

---

## 七、总览之二：漂移轴地图

把四篇的实验摊开之后，会发现一件比任何成功率对比都更重要的事：**它们各自守住了一条不同的漂移轴，而且没有任何一篇同时打下两条。**

更妙的是，它们的失败是互补的。StellaVLA 的固定前缀无法在执行漂移后重新规划，于是长时程归零；WAM-TTT 的快权重在执行时是冻结的，于是处理不了「车顶被人拿走了」；RoboTTT 的场景、相机、本体全程固定，于是从未面对真正的分布漂移；Zero-WAM 从未测过未见环境。

![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1TutbJplSnKVM8X5X2icEFNfcU11NwULTuo9w6Cg6WNz0hP7hRCSBU8cFcHavNBke6mE0lkQqSoGJg7IxI2o9NAfFEjnibJdPOlwjFg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)图 7　四条漂移轴上的覆盖情况。所有格子里的数字来自各自论文，口径不同，不能横向比大小；这张图读的是覆盖与空缺，不是强弱排名。

### 一个此前没被注意到的事实：只有一篇真的在用真实的人类视频

![Image](https://mmbiz.qpic.cn/mmbiz_png/aicNBlf1TutZr92uahs0p4hib7iaubdUuRzib2TO3z00dWWhjyfh1cqTYff6EAO5ibic8T0ZUn1uPg9OCFgBhnB8ZySsXXXwsabAMT9A0NmOibXgjc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)图 8　四篇里只有 WAM-TTT 直面了真实、野外、跨域的人类视频。还有一个结构性巧合值得一提：StellaVLA 和 Zero-WAM 都是用 VLM 把机器人轨迹重新渲染成另一种模态当上下文，一个输出结构化文本，一个输出合成视频。两篇互不引用，却发明了同一个套路的两种变体。

---

## 八、总览之三：三条被反复验证的经验

四篇路线相反，却在三件事上撞到了一起。这三条比任何单篇的成功率都更值得带走。

**一、人类信息只许改写感知或生成侧，不许直接触碰动作侧。** WAM-TTT 把 TTT 残差只加在视频专家、动作流零扰动；Zero-WAM 让人类视频只进视频分支、动作分支条件完全不变。两条完全不同的技术路线（快权重 vs 上下文）得出了同一个结论。RoboTTT 看起来是反例，它把 TTT 加在动作头上，但它的上下文是**机器人自己的**历史而不是人类的，恰好不违反这条。这大概是目前跨本体设计里最稳的一条经验。

**二、训练期挂辅助推理，推理期整个剥离。** StellaVLA 的空间语言专家（保留要 3177 ms，剥掉只要 88 ms，36 倍）和 Zero-WAM 的 IFP 模块，都是「训练时挂上、部署时删掉」。而且两篇都意识到同一件事：**辅助目标必须去监督主分支，不能自成一路**，否则推理时一剥离，主分支什么也没学到。Zero-WAM 对这一点的论证最清晰。

**三、上下文不是免费午餐，模型一定会走捷径。** RoboTTT：多给一帧历史的 GR00T N1.7 Hist. 在 Pup Go Car 上 39.5%，低于无历史版本的 57%。Zero-WAM：没有 IFP 时，加人类视频（28.55）低于不加（39.44）。StellaVLA：训练时用 Image-only 分内更好，但 OOD 差 6.3 个点。三篇在三个完全不同的设定下独立撞进了同一个坑。

---

## 九、总览之四：未走之路

这是我读完四篇后最想留下的一节。下面每一条都是「按现有材料判断，边际信息量高于再发一个新机制」的空缺。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aicNBlf1TutZKzejnrFesa3AENgBIiaxa2VjeJko1r4YMppjf32WSoWtS3FS2YhZiatMaJibkPedQL5He0xuKj80E3iciaWgB8mxPRY3lu9C0S61o/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)图 9　左：把任务时长和分布漂移当两个坐标轴，四篇落在三个角上，右上角空着，而那里正好是家用场景的真实位置。右：四篇各自贡献了一块可复用零件，它们在架构上互不冲突。

1. **三方对照，一次都没人做过。** 在同一 backbone、同一数据、同一评测上并排跑结构化文本上下文、合成视频上下文、快权重记忆。StellaVLA 引用了 WAM-TTT 和 RoboTTT 却一个基线都没做；Zero-WAM 在 §6.2 明确与两者对立却也没比，而且完全没引用 StellaVLA。四篇不同程度的互相知道，但却零对比。
2. **常量前缀加递归状态的叠加。** Zero-WAM 解决了未见任务但长时程仍崩（堆三块积木 9.0%），RoboTTT 解决了长时程但只在固定场景固定任务上。前者的上下文是 episode 常量，后者是逐步变量，一个占输入前缀、一个占递归状态，架构上不冲突。
3. **KVM 损失接到 RoboTTT 上。** RoboTTT 自述局限第二条是「目前用的是通用 MSE 快权重损失，尚未探索面向机器人的 TTT 目标函数」。WAM-TTT 的 KVM 损失恰恰就是这样一个目标函数，还附带了等价于线性注意力的理论论证。两篇同一个月挂出，互补得几乎像是刻意的。
4. **一个真正强的全注意力长上下文基线。** WAM-TTT 的 WAM-ICL 受限于主干本就不是为长上下文 ICL 训练的，RoboTTT 干脆以「解码延迟随上下文线性增长」为由拒做。于是这个方向最核心的争议，也就是「在专门为 ICL 做的大规模预训练下，纯上下文条件到底行不行」，学术侧一次都没被正面测过。
5. **用合成人类视频训练，拿真实人类视频测试。** Zero-WAM 全文未测，而这是它落地时第一个会撞上的墙。
6. **自动标注与质检的质量数字。** StellaVLA 用 Qwen3-VL 做离线标注（而且主干也是 Qwen3-VL，标注噪声与模型归纳偏置同源），Zero-WAM 用 VLM 打语义分和物理分做质检。两篇都没有报告任何通过率、抽检一致性或错误类型分布。「零人工标注成本」的另一面是「零人工质量把关」。
7. **human-robot gap 三种解法的同台比较。** 机制层跨越（KVM 加相位对齐，代价是 2286 对配对 episode）、设定层绕开（同源相机，代价是这个设定在真实部署里不成立）、表示层消解（结构化语言或合成配对，代价是依赖可靠的 VLM 与相机内外参）。三者的成本结构差着数量级，但从没有人在同一个任务上比过。

> **如果只做一件事，做第 1 条。** 它不需要新机制、不需要新数据集，只需要在一个开源主干上把三种上下文接口并排实现一次。这个实验的边际信息量高于再发一篇新方法。

---

## 十、回到起点：ICL 是涌现出来的，还是造出来的？

GEN-1.5 博客的核心姿态是：ICL 能力涌现自 8 个月连续预训练，**无架构改动、无 meta-learning 循环、无辅助目标**。四篇论文的证据全部指向反面。

- **WAM-TTT**：去掉 meta-training 阶段，成绩从 100.0 / 88.9 掉到 9.0 / 0.0，几乎归零。
- **RoboTTT**：去掉 sequence action forcing 就训不出来；register token 单独加在基线上没用，只有配合 TTT 才有 +18%。
- **StellaVLA**：需要一整条离线结构化流水线（尽管它自己的 λ 消融显示，那个语言监督反而在伤 OOD）。
- **Zero-WAM**：在 15,360 GPU 小时的大规模预训练下，没有 IFP 这个反捷径目标，人类视频条件仍然是**净负收益**。

最后一条尤其致命，因为 Zero-WAM 是四篇里**最接近产业博客路线**的那一篇：大规模预训练、纯上下文、零梯度。它用自己的核心消融证明了这条路线需要显式机制。

到目前为止，「ICL 从 scaling 中自然涌现」在学术侧没有任何支持证据，反证倒有四条。

> **但要公平地说。** 没有一篇学术论文达到过 GEN-1.5 或 S1 声称的预训练规模。所以严格讲，四篇证明的是**「在我们能负担的规模上，需要显式机制」**，而不是「在任何规模上都需要」。这个问题只能等有人在同等规模上做一次消融才能定论。在那之前，两边都还没有资格宣布赢。

---

## 十一、文中数字的三个注意

这个方向现在最容易犯的错误，是把 46.2、79、98.8、46.95 并排放进一张表然后排名。这四个数字分别是：未见家庭场景下的部分给分 progress、固定场景下的 rubric 完成分、饱和仿真基准上的成功率、仿真里未见任务的成功率。它们测的根本不是同一件事。

1. **标注泛化的具体定义**：未见场景 / 未见构型 / 未见扰动 / 未见任务，这四个词的难度差着量级。
2. **标注指标类型**：progress 部分给分、rubric 完成分、二元成功率，前两者都会显著高于后者。
3. **标注样本量**：n=10 的格子（三篇的多数消融）必须按较宽误差理解。

顺带一句：产业博客的数字（GEN-1.5 的 59% 与 83%，S1 的 66% 与 96%）与 RoboTTT 的 one-shot 6/10 落在同一区间，这纯属口径巧合，不构成能力等价。真正可比的是证据形式：四篇有基线、有消融、有 rubric，博客有 demo 视频。

---

## 四篇论文

- WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time　arXiv:2607.06988
- RoboTTT: Context Scaling for Robot Policies　arXiv:2607.15275
- StellaVLA: In-Context Structured Demonstration for Generalizable Vision-Language-Action Models　arXiv:2608.11671
- Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization　arXiv:2608.26103

## 重要限定

四篇论文本身**都没有释放代码、权重或数据**，所有定量结论只能采信论文自报。主干（LDA-1B、GR00T N1.7、Qwen3-VL、Wan-2.2）分别开源，理论上可以重搭，但预训练算力（8× H800、16× GB200、15,360 GPU 小时）对多数团队不可及。

本文所有横向表格与图示均为**定性定位**，不构成性能排序。Zero-WAM 一节中「无 IFP 时人类视频为净负收益」（28.55 对 39.44）是按论文自身两张消融图的数字推算，论文未声明这两个变体训练配置完全匹配，也从未把它们并列，方向性结论应成立但严格等价性需作者确认。产业博客的能力主张多以单个 demo 视频佐证，属能力存在性证据，不应表述为已稳定达成的成功率。



---
