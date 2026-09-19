---
title: "[风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第一篇"
author: 多模空间
date: "2026-09-17 10:02:00"
source: "https://mp.weixin.qq.com/s/uKFzE3jyplG7EwbsFG52kw"
---

# [风向标-具身VLA] 一周 VLA 研究趋势简析（2026.08.10-2026.08.16）-第一篇

## 总体情况

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/4ickydTIAhcUmpUahzibnI8jQibJvIdoupUSPUNdz6wOiaEEsdFNSPb9cwbyHDH3ErLWhqCH1pWRbTSl6UmfrQHaFEFwFRCgITwYoE6l6iayTqag/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

## 具体文章

# 架构模块

#### 00. Mamba-based Selective State Space Modeling Improves the Accuracy-Complexity Tradeoff of SmolVLA Vision-Language-Action Experts

- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.21407v1
- • **首发：** 2026-08-10/v1
- • **简介：** 为了轻量化 VLA，该研究采用 Mamba 的选择性状态空间模型替代因果自注意力，并比较不同动作执行长度下的表现；结果显示，该研究基于的 Mamba 结构在逐步规划时保持了与Transformer相近的成功率且参数更少，在连续执行更多动作时也能更好地保留任务成功率，更适合实时部署；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcXvfc1fpmmrySvhKgoEHj5cWCSVQ5aDZAmWBPictF058j0e8ClF2pfQ7IjE4rGVmiaxro2GPE46elCianTAz6ezPEpMRqIVrsuWs4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)
- • **评测：** 模拟：LIBERO
- • **发布：** 哈马德·本·哈利法大学（卡）、卡塔尔大学（卡）、詹姆斯库克大学（澳）

#### 01. VANE: Reliable Test-Time Training for Vision-Language-Action Models via Future Visual Representation Prediction

- • **简称：** VANE
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.09448v1
- • **首发：** 2026-08-10/v1
- • **简介：** 现有 VLA 虽可利用部署阶段的无标注数据进行测试时训练（TTT），但不同任务的修正容易相互干扰，在线更新也可能在效果尚未验证时就改变后续动作，影响闭环控制；这项方法先把修改放在独立的“试用区”，结合当前画面和指令提出调整，再观察执行后的结果，确认有帮助才正式启用，不合适就撤回；测试中它比普通 TTT 整体效果更好，但面对不同任务和机器人时，获得的帮助并不完全相同；![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/4ickydTIAhcWSpQTLYgFOTnQRiaaOqzhjBRnnOLwRp4xYYP2MCA7iadyia8JPiafrHyV1tpRsDkjfN5dkO4ELjDOsibPjkdNVnhw2Go5GpcUZQLDA/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=2)
- • **评测：** 模拟：SimplerEnv-WidowX/Google Robot；
- • **发布：** 香港中文大学（深圳）、北京邮电大学、中国科学院自动化研究所、中国科学院大学、理想汽车

#### 02. Trajectory Divergence Horizon Decision for Reliable Dual-Arm Surgical Subtask Manipulation

- • **简称：** TDHD
- • **领域：** 医疗
- • **链接：** arxiv.org/abs/2608.09125v1
- • **首发：** 2026-08-10/v1
- • **简介：** 现有医疗手术场景下的 VLA 常按固定长度一次执行一段动作，手术环境稍有变化，后续动作仍会照原计划继续，容易累积偏差并带来操作风险；该研究提供的方法会生成两份受到轻微干扰的动作计划，如果两份计划越走越不一样，就说明后面的动作可能不可靠，机器人会提前停下，重新观察画面并规划；这种边做边检查、发现不稳就重来的方式，让真实双臂机器人完成多种针和组织操作时更可靠，尤其能减少任务快结束时的失误；![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/4ickydTIAhcVniafkPdJAFicSZN3IzceOAZUsKibAcw78L7CW1N8W1ibkIQryKo94NwUs3pCqGB0K4tr090FwT3STJEiaBxsIicCNSZciagJiagdN7Qw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)
- • **评测：** 实机：RM65-B 双臂 + 基于 dVRK 改造的末端
- • **发布：** 香港中文大学、华为、深圳河套学院

#### 03. Hermite Curves as Trajectory Priors for Vision-Language-Action Models

- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.01265v2
- • **更新：** 2026-08-09/v2（首发：2026-08-02/v1）
- • **简介：** 一般 VLA 会把一段动作当成许多彼此独立的小指令来预测，导致受控物体运动时可能忽快忽慢，切换到下一段动作时还会突然跳一下；这项研究让模型用一条由起点、终点和两端速度决定的平滑曲线来理解整段动作，并比较直接预测曲线、用曲线打底后修正、只在训练时要求动作接近曲线等做法；结果发现，只在训练阶段用曲线引导最实用，能提高任务成功率，而且部署时不用增加额外计算；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcXTghjZvOYjiaHUPTahjw4RvKr6c9vHK1lywJ8VJLNaYcGFxEVaqhajfYlyhlndTH77hGfU5wgDWpP7ujVANFofcMQPWuM6BBeM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)
- • **评测：** 模拟：LIBERO、LIBERO-Plus；实机： Franka 单臂、Cybopal 单/双臂、ARX 双臂
- • **备注：** aopolin-lv.github.io/Hermite
- • **发布：** 哈尔滨工业大学、江苏细胞壁智能科技有限公司（Jiangsu Cytoderm Intelligent Technology Co., Ltd. ）、新加坡国立大学（新）

#### 04. Cross-View Action Consistency for Camera-Robust Vision-Language-Action Policies

- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.06965v1
- • **首发：** 2026-08-07/v1
- • **简介：** 为了克服 VLA 训推相机视角不一致导致的问题，该研究仅使用场景RGB、语言和本体状态，并屏蔽腕部画面，不依赖相机标注、外参、深度或点云；针对流式VLA，通过同一机器人状态渲染正常与扰动视角，分别进行流匹配监督，并约束两种视角在相同流坐标下预测一致的动作流速度；结果表明，该方法提升了未见相机位置下的仿真与实机成功率，同时保持原视角表现，打乱配对后性能明显下降，说明动作等价视角配对是关键；![Image](https://mmbiz.qpic.cn/mmbiz_jpg/4ickydTIAhcUElG477tS2iahQ0Lh7zBv5QHKnOHwcE9gVOyIlaQqibl6jKBA0m980WGAhMSdhSFDbFSog8YDV400icQzYr0c21UTQ8LTax8DYoY/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=5)
- • **评测：** 模拟：LIBERO-Plus；实机：RealMan RM-75
- • **发布：** 清华大学、阿姆斯特丹大学（荷）、阿姆斯特丹自由大学（荷）

# 分析诊断、空间感知

#### 00. From Recovery to Drop-off: How Action Post-training Reduces a VLM's Late-Layer Depth Decodability

- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.08904v1
- • **首发：** 2026-08-09/v1
- • **简介：** VLM 原本能够从图像中判断物体远近，但训练成 VLA、让它学会输出机器人动作后，这项能力变差了；研究人员逐层检查发现，它并不是只在最后一层出问题，而是每一层都比原来的 VLM 弱，最后几层还会突然下降；相关实验将这一问题定位到后层 MLP，动作训练干扰了其中对深度信息的写入，移除相关写入可恢复大部分末端损失，而注意力模块未表现出相同影响；结果说明，动作后训练可能在整体削弱空间表征的同时，进一步破坏后层对深度信息的保留；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcW5uJxicuX3ibq10M44LDDqrw9xXWbCoL220XMtwicvOCpdb2TMplALfG4QRKql26AQXia7Q3D1bickD4vxcCxF89OX8QhseYKZHH2s/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)
- • **评测：** 模拟：LIBERO
- • **备注：** Accepted to the archival proceedings track of the Embodied Multimodal Reasoning (EMR) Workshop at ECCV 2026
- • **发布：** 纽约大学（美）、Reflex（美）、蒙特利尔大学（加）、马斯特里赫特大学（荷）

# 类 Agent、长程记忆

#### 00. Skills in Weights, Memory in Code: Hybrid Learning for Memory-Dependent Robot Manipulation

- • **简称：** HyMeS
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.09410v1
- • **首发：** 2026-08-10/v1
- • **简介：** 该方法基于 coding agent 针对长程任务场景搭建方案，让 VLA 学习可复用的低层动作，由 coding agent 根据执行反馈学习高层记忆管理规则，并结合本体信号与多帧 VLM 判断任务阶段是否完成，持续更新记忆；实验表明，它只需为基础动作提供示范，就能组合完成多种依赖长期记忆的任务，整体表现优于对比方法；
- ![Image](https://mmbiz.qpic.cn/mmbiz_jpg/4ickydTIAhcX4rOgRSnxqGzxPlkjfic1eLADEhtkNlCrqBS7dfMI2f1C1GBYUlo9D5keDxRQuDd8Jibk0KxyNQkibEgDgZHqlXL1kkE0bfsWick4/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=7)

- • **评测：** 模拟：RoboMemArena；实机：LeRobot SO 101
- • **发布：** 西北大学（美）、明尼苏达大学（美）、斯坦福大学（美）

# 长程记忆

#### 00. OnEvoMemory: Evolving Memory through Online Robot Rollouts for Pretrained Robot Policies

- • **简称：** OnEvoMemory
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.08749v1
- • **首发：** 2026-08-09/v1
- • **简介：** 机器人做很长的任务时，容易忘记前面做过什么，比如已经放好的物体又去拿一次；该方法为预训练机器人策略加入价值引导的记忆模块，同时保存近期信息、高价值经历和重要状态变化，并根据任务结果学习记忆取舍；离线示范用于建立初始记忆，在线成功与失败轨迹进一步调整选择方式，从而帮助策略识别任务阶段、避免重复操作，并提升基础VLA的长时序任务表现；![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/4ickydTIAhcXExE0QCCATpicdQMDDYhGAjeQF5XDxVhKkurlKRxefQZI0A1pf2H7WNGqZNMn4qeN5oQCwo4vD7R7BvPibXoWJllhKiaJsibcicdXE/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=8)
- • **评测：** 模拟：LiberoLong-10、RMBench（选了两个长程任务）
- • **备注：** Accepted as a poster at the ECCV 2026 Workshop on Embodied Multimodal Reasoning in Physical Environments (EMR)
- • **发布：** 上海交通大学、清华大学

# 异常处理

#### 00. WA-SpecDec: World-Aware Speculative Decoding for Vision-Language-Action Models

- • **简称：** WA-SpecDec
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.08725v1
- • **首发：** 2026-08-09/v1
- • **简介：** 传统 VLA 每生成一步动作都要反复计算，而已有加速方法会先猜出一串动作，再让主模型统一检查，但以前无论机械臂在空中移动，还是已经靠近物体，都使用同一套误差标准，因此精细抓取时更容易出错；现在模型会先判断周围环境和接触风险，再用这些信息辅助猜动作和检查动作；这样一次可以通过更多安全动作，在基本保持任务效果的同时执行得更快，也更少在接近物体时发生碰撞或抓偏；![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/4ickydTIAhcW1MZJ1HZbhmGwTcbCujserTFXBSMkxicUjz9bSCicTuEqHJQojBsW6xTRVUXXW9rcbDkpKN6ldOPd6APpqAttkR3GibTB2VuxSsY/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=9)
- • **评测：** 模拟：LIBERO、SIMPLER-Env
- • **评测：** 模拟：LIBERO、SIMPLER-Env）

- • **发布：** 悉尼大学（澳）

# 性能提升

#### 00. Depth-Wise Probing and Pruning of the Planning Token in a Driving Vision-Language-Action Model

- • **领域：** 智驾
- • **链接：** arxiv.org/abs/2608.07361v1
- • **首发：** 2026-08-07/v1
- • **简介：** 智驾场景下 VLA 通常让动作决策经过很深的语言模型，但轨迹规划是否需要全部网络层并不明确；研究逐层读取同一个规划 token，检查导航意图何时形成，以及轨迹信息何时能被原有规划器正确使用；结果发现，导航意图和规划信息在浅层已经出现，但还未整理成规划器熟悉的表示，后续网络层主要负责逐步对齐；利用学习式读出可提前提取这些信息，移除部分影响较小的层也能提升解码速度，且未发现明确的特定任务类别退化；结论仅适用于所测试的模型和评测设置；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcWqPB9TJKb9HtvH50zAWhBdRICMicHXtYnwH6CUKntRYfww3OicpuZHlemsYrgGIdaJqrfURmYIuHgg1NQHUCtpJWh58h6FbE9C8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)
- • **评测：** Bench2Drive
- • **备注：** Accepted at the 6th DriveX Workshop (Foundation Models for Autonomous Driving), ECCV 2026
- • **发布：** 博世（Robert Bosch GmbH，德）、卡尔斯鲁厄理工学院（KIT，德）

# 训练范式

#### 00. TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models

- • **简称：** TEMPO
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.07314v1
- • **首发：** 2026-08-07/v1
- • **简介：** VLA 往往通过 SFT 适配任务时容易遇到训练与实际执行分布不一致的问题，而现有 RL 一般会统一更新所有模块，忽略语义理解与动作控制的不同作用，可能破坏原有能力；该方法冻结预训练视觉语言主干，仅分别优化语义投影层和动作专家，前者低频更新以稳定高层语义，后者高频更新以快速吸收在线控制反馈；实验表明，这种双时间尺度训练在模拟与实机操作中均优于预训练模型和常规 RL 后训练方案，并能保持更高的评测回报；![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/4ickydTIAhcUADvFogNeJQj0XicPRkWvYCCeJia72wqus18kV4HBsHoicENlKnWTeXhRhqQPZTb4y4KdNK5s95wMZTjB7ViaCibxfw5zjKE4KFLps/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)
- • **评测：** 模拟：CALVIN ABC→D；实机：Synria Alicia-D
- • **备注：** anonymous.4open.science/w/tempo-page
- • **发布：** 浙江工商大学、皇家理工学院（KTH，瑞典）

#### 01. RecoverFly: A Failure-Aware Reinforcement Learning Post-Training Framework for Aerial Vision-Language Navigation

- • **简称：** RecoverFly
- • **领域：** 导航
- • **链接：** arxiv.org/abs/2608.09467v1
- • **首发：** 2026-08-10/v1
- • **简介：** 为了同时应对 VLN 中，基于行为克隆难以在闭环执行中提供有效纠错，采用 RL 又面临样本利用率低、长尾场景覆盖不足和策略训练偏移等两个方法问题；该方法采用面向失败的 RL 后训练，通过适配受动作语法约束的逐 token 优化、反复学习未解决的失败案例，并结合分阶段长尾场景训练与参考策略约束，提高纠错和场景适应能力，同时减少已有能力退化；实验表明，其在已见和未见环境中均取得更好的导航表现，并提升了鲁棒性与泛化能力；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcVAsPQhszRZrfic8utWicic60VfdZFoJMz1OKkmrwhWfzYjvKcBz47xf4QhhJ8WwOY0CeCVssvRqZibYzicAHugtozQJdP64ZYoHwVw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)
- • **评测：** TravelUAV
- • **发布：** 吉林大学、清华大学、北京航空航天大学、北京中关村学院

# 空间感知

#### 00. WNM-3D: A World Navigation Model with 3D Scene Conditioning for Closed-Loop VLN

- • **简称：** WNM-3D
- • **领域：** 导航
- • **链接：** arxiv.org/abs/2608.07267v1
- • **首发：** 2026-08-07/v1
- • **简介：** 现有连续式 VLN 虽能理解图像和指令，却主要直接预测动作，没有显式建模移动后画面应如何变化；已有 WAM 可联合生成未来视角与动作，但缺少从历史观测中提取的三维几何条件；该方法用冻结的几何编码器整合单目视觉历史，将三维场景信息转成固定长度 token，持续指导未来画面与动作生成，并结合示范学习、策略访问状态适配和闭环强化优化；实验表明，其闭环导航优于强 VLM 策略和二维条件版本，未来画面与动作更一致，视觉运动误差更低；![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/4ickydTIAhcUHsD7odXkdPemgOfjlibwqjf0oP8X3gDylsib2L6W9T62C0fORtCjUrIzpZKR58ibZV5DdW84QFFUofKiaOSq9AEyfUpSPrTYhFTI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)
- • **评测：** 模拟：GN-Bench
- • **发布：** 中国电信人工智能研究院（TeleAI）、浙江大学、同济大学、上海交通大学

#### 01. AnyCamVLA: Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action Models

- • **简称：** AnyCamVLA
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2603.05868v2
- • **更新：** 2026-08-10/v2（首发：2026-03-06/v1）
- • **简介：** 为了应对训推视角差异导致 VLA 表现下降的问题，该方法利用前馈式新视角合成模型，在测试时实时将当前画面转换为接近训练相机配置的图像，无需新增示范数据、继续微调或修改网络结构，可直接接入基于 RGB 的策略；实验表明，该方法优于依赖数据增强微调或额外 3D 特征的方案，并能提升真实机器人面对不同相机参数和移动相机时的视角鲁棒性；![Image](https://mmbiz.qpic.cn/mmbiz_jpg/4ickydTIAhcXUrGbZ6zXpaWrYOp24UwBoYCiayFeLaN934dzkUqxdmDgmKER5tfjgjQP5lelX7pxI8mtOhoKVXbfzgyCJbCS5BQdVndDwepCA/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=14)

- • **评测：** 模拟：LIBERO、LIBERO-Plus；实机：Franka Panda
- • **备注：** Accepted to IROS 2026；heo0224.github.io/AnyCamVLA
- • **发布：** 首尔大学（韩）、麻省理工学院（美）

# 世界模型

#### 00. WAM-Diff2: Hierarchical AR-to-Diffusion Distillation for Highly Efficient Autonomous Driving VLA

- • **简称：** WAM-Diff
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.01035v2
- • **更新：** 2026-08-07/v2（首发：2026-08-02/v1）
- • **简介：** 现有端到端自动驾驶 VLA 无论是依赖自回归逐步生成，还是基于扩散策略都有各自的问题；该方法通过分块适配、分块蒸馏和跨尺度整模蒸馏，将预训练自回归模型逐步转换为离散扩散 VLA，在保留语义基础的同时实现并行解码；实验显示其减轻了暴露偏差，理解、感知和规划表现接近自回归基线，并提高了解码效率；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcXs2GGgxZcFIaGbNYcc3gIMYgjbxMgr41pVhSLbofWibh1VkUcaKZcCVFTXWu2Qgs7POibibYOzKiclB9DEwF30pW2EAZRsJOicz5as/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)
- • **评测：** NAVSIM、Bench2Drive、LingoQA、DriveBench、COCO
- • **备注：** Copyright © 2027, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.（文章模板采用）
- • **发布：** 复旦大学、引望智能技术有限公司（Yinwang Inttelligent Technology Co.,Ltd）

#### 01. SLIM-0.5B: Learning Action-Grounded Predictive Latents for Robot Manipulation

- • **简称：** SLIM-0.5B
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.09771v1
- • **首发：** 2026-08-10/v1
- • **简介：** 现有 VLA 每个控制步骤都依赖多模态大模型完成感知、语言理解和动作生成，其中许多能力面向开放语义，对连续机器人控制而言成本较高；像素级 WAM 还需预测大量与控制无关的画面细节；该方法用轻量模型学习与动作相关的潜在表示，既预测动作带来的未来变化，也根据观察变化还原动作，并结合自监督轨迹遮盖训练与流匹配生成语言条件动作；仿真与实机结果表明，其表现达到或超过多种大型 VLA 和 WAM，同时参数更少，无需额外具身预训练，推理延迟与显存占用更低；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcWiaoI5b5Ve5x37tBgAFiahzbApHjJxsAOGicRm9uRfeYCI4TjgYfz35JbxNrgdxUTY32HGbdnOYR7Pic8K3gK3x3Or0j28icWLdb04/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)
- • **评测：** 模拟：LIBERO、LIBERO-Plus、CALVIN ABC→D；实机：单臂
- • **备注：** kzz1031.github.io/slim-project-page
- • **发布：** 复旦大学、北京智源人工智能研究院（BAAI）、清华大学、中国人民大学

#### 02. World Tokens: Enhancing Embodied Policies with Training-Time World Modeling

- • **简称：** World Tokens
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.09730v1
- • **首发：** 2026-08-10/v1
- • **简介：** 一般 WAM 虽能学习物理过程，但在控制时保留视频生成模块会增加推理开销；该方法在训练中将 VLM 特征压缩为统一的世界 token，同时用于预测未来画面和生成动作，使动作表征受到环境变化监督；部署时移除世界模型分支，仅保留 VLM、适配模块和动作专家；实验中在多项仿真任务上表现出较强竞争力，实机成功率优于仅训练动作的基线，同时保持接近常规 VLA 的推理延迟；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcVsZUEs9yZEib3OqePaQib34hLxD59wlCgTZu09OkwoApXBHA8eDzBtUMNPhhzFAZBnVIjhM70Oj7Qa8WVaOwhibOSdNJK9icjBBWA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)
- • **评测：** 模拟：LIBERO、SIMPLER；实机：Galaxea R1 Pro
- • **发布：** 中国移动九天研究院、北京中关村学院- **评测：** 仿真：LIBERO、LIBERO-Plus；实机：Franka Panda
- • **备注：** Accepted to IROS 2026；heo0224.github.io/AnyCamVLA
- • **发布：** 首尔国立大学（韩）、麻省理工学院（美）

#### 03. -0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation

- • **简称：** ω-0
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.06375v2
- • **更新：** 2026-08-09/v2（首发：2026-08-06/v1）
- • **简介：** 过去的人形机器人通常先管走路，再管伸手拿东西，各部分各做各的，边走边拿时容易不协调；这个方法让模型同时看任务要求、摄像头画面和身体状态，提前判断接下来场景会怎样变化，再一次性安排全身动作，不需要生成完整的未来视频；因此机器人可以一边移动和保持平衡，一边调整身体并操作物体，在多种真实家庭任务中表现得更连贯稳定；![Image](https://mmbiz.qpic.cn/mmbiz_jpg/4ickydTIAhcVs6z5wSwRlkwwEbhJRHUEGbRnt3X79tBFraA6CicdjB6L6OMG5r4jHdVIJEQVDUiaKGkO4VtEicoDkxc6BYfeBOGcCzjaHflp0xA/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=18)
- • **评测：** 实机：人形机器人 + Inspire DexHands 灵巧手
- • **备注：** gentlefress.github.io/OMEGA-0\_page
- • **发布：** 南洋理工大学（新）、北京大学、北京智源人工智能研究院（BAAI）、香港科技大学（广州）

#### 04. JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling

- • **简称：** JEPA-WAM
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2608.09381v1
- • **首发：** 2026-08-10/v1
- • **简介：** 以往一些 WAM 会先生成机器人执行动作后的未来画面，再据此决定怎么动，但这种方式计算量大、部署较慢；该方法在预训练 V-JEPA 潜空间中，通过共享预测器同时学习当前到未来的视觉变化与连续动作，保留空间结构和图像块之间的对应关系，并能接入已有 VLA 而不改变原有感知与动作通路；实验表明，其在缺少大规模机器人策略预训练时也取得较好效果，接入预训练 VLA 后表现进一步提升，并能适应视觉和空间变化，在真实双臂操作中展现出较好的泛化能力；![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/4ickydTIAhcVibdyDaoNnaiaP2dTwgMrHhckGyyT68cAGCCmAUrJzU3Yz2FrBeOE7zTLN8zCIntibTAezJS22DpzjX6FDRxAlD6OQwticGuHcY5k/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=19)
- • **评测：** 模拟：LIBERO、LIBERO-Plus、RoboTwin 2.0；实机：AgileX COBOT Magic 双臂
- • **备注：** spritewithoutice.github.io/JEPA\_WAM
- • **发布：** 中国人民大学、星源智机器人（XYZ Embodied AI）、数据工程与知识工程教育部重点实验室、数据库与商务智能教育部工程研究中心、深圳理工大学、清华大学

# 类 Agent、数采评估

#### 00. ActiveFly-Bench: Aligning Embodied Question Answering with Vision-Language-Action for Aerial Embodied Perception

- • **简称：** ActiveFly、ActiveFly-Bench
- • **领域：** 通用
- • **链接：** arxiv.org/abs/2607.10180v2
- • **更新：** 2026-08-10/v2（首发：2026-07-11/v1）
- • **简介：** 该研究将无人机导航场景下的任务理解、行为规划与精细控制连接起来进行评测；该工作将空中主动感知拆分为问答、观察行为规划和语言引导控制三个层级，使用真实与仿真户外数据，并构建结合视觉语言推理和闭环控制的无人机智能体，在实机上进行验证；实验发现，现有模型在行为规划、视角调整和稳定执行方面仍有不足，为研究空中具身智能提供了新的评测基础；![Image](https://mmbiz.qpic.cn/mmbiz_jpg/4ickydTIAhcXUrI24mjNGuMys31o0Lic0iaESrSia02XKPJbS9s8YdHpgkZSiaF2t1EvFI1TI32WkCwoEeGPOGYBNWDBNibHd1tq0TayATlF0q2icQ/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=20)![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/4ickydTIAhcVIOG4sLumycuhicROzx8hjSpS4IvEgSDWGLn3KPSQQtt2E2Hia24ZDS17PnWf5n2TVicRicYMOM1NNPMc5PBOu6sBAo0nNAAtCjbY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=21)
- • **评测：** ActiveFly-Bench（自建）
- • **备注：** lvmolvmo.github.io/ActiveFly
- • **发布：** 清华大学、北京流形空间科技有限公司（Manifold AI）

# 架构模块、数采评估

#### 00. CMU-Drive and V2V-VLA: Cooperative Multi-agent Unified Driving with Reasoning Benchmark and Vehicle-to-Vehicle Vision-Language-Action Models

- • **简称：** CMU-Drive、V2V-VLA
- • **领域：** 智驾
- • **链接：** arxiv.org/abs/2608.07621v1
- • **首发：** 2026-08-07/v1
- • **简介：** 现有智驾场景下 VLA 多面向单车独立决策，难以让多辆联网车辆共享感知信息并协同推理、规划，在复杂交通中也缺少统一的闭环评测方式；该研究构建多车协作的端到端闭环基准，并让模型在一次前向计算中同时生成驾驶动作、未来轨迹点、语言推理和通信策略；实验建立了协作式 VLA 驾驶的基准与基础模型，为多智能体闭环协同驾驶研究提供了统一起点；![Image](https://mmbiz.qpic.cn/mmbiz_png/4ickydTIAhcVtiaibnuiaafjpsvLUZsYiafY4J6JMRkkvQ9L8hRXMJtsNCvTHn9Lg4fXS3T4pFtibZWYdMLuxJsgSW8GX8pYAibJlicsibmHT4cKgSw8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=22)
- • **评测：** CMU-Drive（自建）
- • **发布：** 卡内基梅隆大学（美）

备注：本文所涉论文解析，仅针对本文发布时arxiv平台已公开的被解析论文对应版本作出；文中涉及原论文的图表、数据均引用自原论文，相关知识产权归原权利人所有，如涉侵权请联系删除；本文数据汇总与观点解读均为个人基于上述论文解析系列的独立理解与统计，仅代表个人观点，非原作者或相关机构的官方认定，受认知局限难免有错漏，如有必要请联系修改；本文仅作学术交流参考，无法替代原论文，深入研究请查阅原文；
