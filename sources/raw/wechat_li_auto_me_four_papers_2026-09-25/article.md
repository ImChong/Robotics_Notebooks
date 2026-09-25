---
title: 理想一口气连发四项具身基座模型工作：车企进具身，理想第一次交卷。
author: 具身智能研究室
date: "2026-09-25 09:52:40"
source: "https://mp.weixin.qq.com/s/UVSRMDa8Aq2oJtqkRUU_EA"
---

# 理想一口气连发四项具身基座模型工作：车企进具身，理想第一次交卷。

人形机器人运动控制知识库已开源

我把人形机器人运动控制知识库开源了，从重定向一路整理到世界模型。面向初学者、求职者、算法工程师、科研人员和技术负责人。

结合 Agent 使用效果会更好：

https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main

论文信息

理想汽车基础模型团队

01ME-Brain-1.0: Memory, Cognition and Action for Evolving Embodied Intelligence02ME-VLM: A Unified VLM for Embodied Cognition and Agent Coordination03MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence04ME-Dex 1.0: Bringing Heterogeneous Tactile Sensing into World Action Modeling

机器人会叠碗了，为什么还是插不好充电器？理想汽车在 ME-Brain 1.0 的真机测试中，让一台 Piper 双臂机器人分别尝试两项任务：叠碗十次全成功，插充电器十次只成功一次。

叠碗和插充电器都要看准、动手，后者还多了插入时的对位和接触。机器人做这些事，需要知道目标在哪、手该怎么走、碰到物体后发生了什么；失败的那次也不能白做。理想汽车这四篇论文，分别研究经验怎么留下、下一步怎么判断、动作怎么生成、触觉怎么用。

## 记忆与演进

ME-Brain 1.0 关心机器人做完一件事后留下些什么。一次抓取的画面、手臂位置和结果，原本只是这次运行的记录。它想把记录整理成下次做事时能查、能用的经验。

机器人执行时，记忆模块保存画面、关节状态、动作和结果。长长一段过程会被整理成抓取、放置等事件，再归纳出哪些做法成功、哪里容易失败。事件还连着原始记录；以后查到一次“抓取失败”，就能回看当时的画面和动作。

有了这些记录，认知核心才能在新任务里找相关经验。用户让机器人把两个物体依次放进篮子，它要确定顺序、选择技能，放完一个还得检查有没有放稳。如果物体掉了，就根据刚收到的反馈改计划。

确定下一步之后，动作模型生成关节和夹爪动作。抓取、碰到物体、松手放置，这些时刻最容易影响结果，模型会重点处理这些时刻附近的画面和相关历史。新动作产生新记录，下一次又能从记忆里查出来。

![ME-Brain 1.0 论文图 1：记忆、认知、动作组成的执行与经验循环](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXdlszdloBAO1c0vJkibs5W6FTgwWwjgZgErxk9lnLNU3QjiaaeDrPeorZLYyZ5mzWqZiax4OicLXZNUUibkeIGianrIok4kI9vb9oVs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

论文把这个过程叫“自我演进”。更新的是外部记忆和技能经验，后续任务可以继续调用；模型权重不会随着每次任务自动改变。

开头的叠碗和插充电器，来自 Piper 双臂机器人的真机测试。六项任务各做 10 次，平均成功率 66.7%。论文也单独测了动作模型：在 RoboMME 的计数、物体持续性、指代和模仿四类测试中，总体平均成功率 47.88%；在 RoboDojo 仿真任务中为 16.03%。这些成绩说明系统已经能完成一些操作，也显示插接这样的精细动作仍然很难。至于那次插接失败具体卡在哪里，论文没有给出单独的故障分析。

## 认知与规划

接下来是 ME-VLM。桌上有几个相似的物体，机器人要找准用户指的那个，决定先抓哪里、放到哪里，动作结束后还要确认有没有放对。ME-VLM 研究的是这一串判断。

这篇论文还训练模型处理数字任务，比如拆解目标、调用工具、读懂返回结果。它与 ME-Brain 的联系很直接：ME-Brain 讲认知模型怎样和记忆、动作部分一起工作；ME-VLM 讲这个认知模型怎样训练出来。

![ME-VLM 论文图 2：物理世界与数字任务中的感知、推理、行动和检查](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVAzISicTyPV5aZB6xKOqbiaeOXfaNqek8gJ7ZYS0pjoSAPic5EtUyrwHsoErBtvhvmdaJPmXV91CcaFicBiawbrmttR4SyRM62qGc8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

训练从一个多模态模型开始。作者先补上与行动有关的知识：空间关系、可操作位置、动作前提和执行结果；接着让模型练习根据当前状态选择下一步，以及怎样检查这一步是否完成。有了这个基础，再分两条强化学习路线：具身专家侧重物理约束、空间判断和执行反馈；数字 Agent 专家侧重任务拆解、工具调用和多轮交互。最后通过多教师蒸馏，把两类能力收进一个部署模型。先分开练、再合到一起，是为了兼顾两种不同的任务要求。

蒸馏时，学生模型会走自己的任务轨迹，再由专家指导它在这些状态下如何决策。机器人真执行时总会遇到示范里没出现过的局面，这时它还得根据眼前的新情况继续判断、修正动作。

![ME-VLM 论文图 4：具身训练、双专家强化学习与最终蒸馏](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXzatQNkl2K3ptQvedshlgMoHRuFiaVGsqohbmNvia5s8HibrxfXaibibvyBiaIibIUaiaAdkZKlURiaqz0SUhgOia7b3OFb9nKz7AnS2mYQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

论文里有个仿真堆叠案例：模型第一次认错了目标，收到执行反馈后重新找目标、改计划，到第三轮才完成。它最有意思的地方，是做错之后还能接着看、接着改。这也解释了训练为什么强调结果检查和失败恢复。

论文发布了 4B 和 35B-A3B 两个版本。较大版本在具身基准和 Agent 基准中的平均分分别为 70.9 和 72.5。这些数字是多项基准的综合分；上面的堆叠是仿真案例，真机案例也没有给出大样本成功率。ME-VLM 与 ME-Brain 使用同一系统背景，相关案例不能重复算成两套独立的真机验证。

## 理解与生成

ME-VLM 讨论怎样选定下一步；再往下，机器人要把这个决定变成具体动作。“把纸巾放进盒子”听起来只有一步，实际要先确定当前该抓什么、从哪里接触，再生成手臂动作，并估计纸巾和盒子会怎样变化。ME-U0 把任务理解、未来视觉预测和动作生成放进同一个模型。

它先用理解专家回答两个问题：现在进行到哪个子任务？应该碰物体的哪里？一个总任务可以持续很久，眼下的目标却会随着场景变化——先抓起物体，再把它放到指定位置。生成专家接过这些信息，联合生成未来视觉状态和机器人动作。于是，动作有了明确的任务目标，未来画面又提供了对动作后果的预测。

未来视觉可以是 RGB 画面，也可以是深度、表面法线或光流，分别提供外观、几何和运动信息。理解专家通过文本预测子任务和交互位置；生成专家用流匹配联合学习视觉未来与动作。机器人动作比未来画面的采样更密，模型因此用多速率位置编码对齐两者的时间顺序，让某段动作与对应的场景变化关联起来。

![ME-U0 论文图 6：理解专家与生成专家共享信息的模型结构](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXw02e5icU945WSzPMtvaw7YVicKRhckbF7p91ng72M5zOUpyVjRP51o9xY4h4hIGT4IHJDkVzkWnCOqVibYqWSe24ic5V5IqQ7UiaE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

ME-U0 用约 4,200 小时整理后的机器人和第一视角示范数据预训练。下游适配后，它在 LIBERO 的 2,000 次仿真测试中达到 99.0% 成功率。换到加入视角、光照、语言等扰动的 LIBERO-Plus，同一个模型得到 82.5%；单看相机视角变化，成功率是 70.2%。平常能做成，视角和环境条件一变还能不能做稳，是另一道题。标准 LIBERO 已有多种方法接近满分，论文表中的 OpenWAM-α 为 99.3%。表中其他方法的成绩来自已发表报告，作者没有在同一环境下重新运行这些基线。

真机部分展示了拿记号笔、把纸巾放进盒子等操作，主要提供定性观察，没有大样本成功率。这些硬件平台也出现在预训练数据中。另一个短板是记忆：模型没有显式保留较长的观察历史，在 RoboDojo 的记忆维度上成功率为 7.00%。回看 ME-Brain 就能看出两篇的分工：ME-U0 让当前理解与未来生成协同，ME-Brain 则把过去的经历引入当前决策。

## 触觉与动作

ME-U0 用未来画面辅助动作生成。可一旦夹爪碰到物体，画面很难完整反映接触力和抓握是否稳定；插接、夹持和双手协作尤其依赖这些信息。ME-Dex 1.0 往世界与动作预测里加入触觉：生成动作时，也预测接下来会“摸到”什么。

加入触觉先要跨过硬件差异：夹爪可能只有几块接触面，灵巧手则在多根手指和掌面布置传感器，覆盖范围与分辨率都不同。ME-Dex 用标准手部区域模板和共享触觉编码器建立对应关系，让不同设备的数据进入同一套表示。为了补充成对的视觉、动作、触觉数据，作者还在仿真中重放已有轨迹，记录力传感器的读数。

有了统一的触觉表示，模型再把视频、触觉和动作交给三个专家分别处理。它们在网络中间层交换信息，动作专家因此可以参考预计会看到什么、会碰到什么。未来画面、未来触觉和动作在同一次训练中联合预测，触觉既是当前观察，也是需要模型学习的未来状态。

![ME-Dex 1.0 论文图 2：视频、动作、触觉三专家及统一触觉编码](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWUjyK2G6wJuaM9ibOXgBtYcz2qg5kJPTWsFvqJv2S8XWqtLbkldNice0fQR3XFmRVvkXxLFBPb8146QdYblZwBVRzXEB4NcJUe8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

在 RoboTwin 仿真基准中，它在 Clean 和 Random 设置下分别达到 91.56% 和 91.92%；DexJoCo 的平均成功率为 65.3%，对比方法 DECO.p 为 57.8%；ManiFeel 四项任务平均为 70%，基线为 55%。各任务的提升并不一致，ManiFeel 的齿轮装配就是一个下降的例子。

最让人意外的是，完整模型在 RoboTwin 测试时把当前触觉输入置零，成功率仍有 91.74%，几乎追上使用当前触觉时的 91.92%。再看训练消融：在 Random 设置下，只把当前触觉作为输入的版本得到 89.54%；加入未来触觉预测后升至 90.60%，再加上三个专家的中层信息交互达到 91.92%。这组结果提示，模型从预测未来触觉中学到的东西很重要；实时触觉输入在这里贡献了多少，还需要继续查。

真机部分展示了不同硬件上的操作，尚无对应的大规模成功率。当前触觉反馈主要用在动作片段之间的重新规划，动作执行中的高频触觉力控还属于后续工作。

![元泽个人名片](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWs8SRbbbnichE26UiaUupdGHJsrFprZIh3YnuHSo1UCEKOTd0O15IwBK01L3Be4aevyF8OsIcYtrSCic7zluocaLWdiaBicn7dvG48/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)
