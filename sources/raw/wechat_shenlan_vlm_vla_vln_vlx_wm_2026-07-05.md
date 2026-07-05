---
title: 五大具身模型详解：VLM、VLA、VLN、VLX、世界模型
author: 深蓝具身智能
date: "2026-07-05 10:56:00"
source: "https://mp.weixin.qq.com/s/xj-rc6v64Ge6onoUPvkHLg"
---

# 五大具身模型详解：VLM、VLA、VLN、VLX、世界模型

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGeDxZns2AFbbmtYvXJ6NdWEQiaYsRAwFvTrR682dL6VTlW7UXCB3tlFcMzSKxacCveABQjQkrWS0yWmnynV8F55bmvZKoIhiayk8/640?wx_fmt=png&from=appmsg#imgIndex=1)

2026年再看具身智能：五类模型正在被“吞噬”，理解底层逻辑才有入场券

——全景架构

相信你不止一次被VLM、VLN、VLA、VLX、世界模型这一串缩写搞得头晕眼花过。

它们分别解决什么问题？彼此之间是什么关系？更重要的是，当几乎所有技术报告都在谈论“多模态融合”与“端到端一体化”时，逐一厘清这些概念还有何现实意义？

所以，本文打算从统一的神经网络底座出发，沿着“感知→导航→执行→融合”的递进脉络，逐层拆解VLM、VLN、VLA、VLX、WM每一类模型的输入输出、运行机制与能力边界。我

用一套清晰的递进关系串联起这个看似复杂的术语群。

**我们开设此账号，除了想要向各位对【具身智能】感兴趣的人传递前沿权威的知识讯息外，也想和大家一起见证它到底是泡沫还是又一场热浪？****欢迎关注****【深蓝具身智能】**👇![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=2)

## 通用底层原理：五类模型的同源技术根基

## **当下具身智能领域主流的VLM、VLN、VLA、VLX、世界模型五大类技术模块，整体技术体系均构建在人工神经网络的通用计算逻辑之上。**

## **在数字化技术的支撑下，物理世界的各类实体信息、空间信息、运动信息、时序变化信息，都可以通过量化、编码、张量化的方式转化为计算机可识别的数字矩阵与特征向量。**

## **所有能够完成数字化建模的客观规律与场景状态，都可以纳入神经网络的学习与推理范围。**

## **这也是五类具身智能模型能够实现场景适配、任务推理、智能交互的通用底层基础。**

## 了解最基本的神经网络，可以通过下图多层感知机MLP。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/uwFbeBKoFGeMBhc9c2P7NDKf3icWib7kECsmxSmkFc0NUT6W0yruFGmz7kyGZvV2rd7qnIbvLLdZcIESbTv2KlcFShqjXbLekrI6yPboTkVtM/640?wx_fmt=gif&from=appmsg#imgIndex=3)

多模态混合编码是贯穿五类模型的通用技术流程，也是不同功能模块能够互通互联、特征复用的关键支撑。

在具身智能任务场景中，环境信息呈现出明显的异构特征：

二维图像画面、自然语言指令、三维空间深度、机器人运动姿态、环境时序演变属于完全不同的信息维度，无法直接完成特征融合与逻辑关联。

多模态混合编码技术可以对各类异构信息进行标准化处理，通过专属编码器完成单模态特征提取，将图像、文本、空间、运动、时序五类信息统一转换为相同维度、相同表征逻辑的离散数字词元与隐向量特征。

统一后的特征数据会被映射至共享隐向量空间，完成跨模态特征的对齐、融合与关联建模，彻底打破不同信息类型之间的技术壁垒，为多模型协同工作、联合训练、特征迁移提供统一的数据底座。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGdPz6pxGj2lldIWH2uZ23Byb2Kiaicpn3mMT16YIiaAzt60KlhWusPqTX6rmGpBAqL3TnibdXa1UHPyiaAXl3p4GYeiabydvSJxpAn6Q/640?wx_fmt=jpeg&from=appmsg#imgIndex=4)

从网络架构与计算逻辑层面分析：

五类具身智能模型不存在本质性的技术割裂，整体均采用Transformer主干网络作为基础骨架，依托自注意力机制完成全局特征关联建模，遵循统一的序列建模与梯度下降训练逻辑。

各类模型的差异化表现，全部来源于后天任务定制化设计：

主要集中在输入数据的模态组合配比、输出特征的任务拟合方向、模型推理的运行场景、时序建模的时间跨度、网络分支的功能配置五个维度。

统一的底层技术体系，让五类模型具备天然的共生属性，能够按照感知、导航、执行、融合、推演的能力层级完成递进式组合，搭建出完整的具身智能技术链路。

这里需要做一个关键区分：

世界模型的底层技术逻辑与其余四类模型保持完全一致，同样依托神经网络数据拟合与多模态编码机制完成建模工作。

两类体系（VL系列 vs WM）的区分主要体现在模型的运行定位与任务使用方式：

其余四类模型（VLM、VLN、VLA、VLX）的推理输出会直接对接硬件设备或程序系统，完成真实场景的感知解析、路径规划与动作执行。

世界模型的推理输出不参与真实设备的即时执行工作，主要用于构建动态虚拟仿真场景，为各类具身模型提供可交互的虚拟试验空间。

明确了底层共性之后，我们逐一拆解每一类模型。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=5)

## **各类模型原理、技术细节与差异化特征**

基于统一的技术底座，五类模型通过差异化的模态配置、网络分支、任务目标，形成了各司其职、层层递进的功能体系。

下面逐一拆解各模型的核心定义、输入输出、运行逻辑与专属差异化特征，清晰区分各模型的功能边界与应用场景。

### **VLM 视觉语言模型（Vision Language Model）**

Vision Language Mode定义为依托视觉感知与自然语言语义协同工作的智能学习模型。

更专业一些来说，指代一类融合二维视觉感知与自然语言理解的跨模态基础网络架构，是具身智能体系中所有图文认知任务的通用载体。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdWm0c3iajo0VEDicXyqFCeKtEnkEib3VqufE1JBkJLU5icYLJVkz3EerrN1teHRpicJlnTWpL4RpuQoSoRXJ5yA3Xl7PBdwyRKwfM4/640?wx_fmt=png&from=appmsg#imgIndex=6)

VLM是具身智能体系中专注跨模态感知理解的基础网络模块，承担整个智能系统的环境认知与指令解析工作，是所有下游导航、执行、推演模型的前置感知底座。

- 输入

VLM的输入数据体系由视觉模态、语言模态双向构成。

视觉模态包含实时摄像头采集的二维RGB图像、场景静态画面、动态帧序列画面；

语言模态包含人类自然语言指令、场景描述文本、任务约束文本等标准化文字信息。

两类模态的原始数据不具备直接融合的条件，需要经过专属编码模块完成预处理，转化为统一维度的特征词元后，再送入主干网络完成特征交互。

- 输出

VLM的输出数据以结构化语义特征与文本信息为主，涵盖场景内物体类别识别、物体空间关联关系解析、场景整体属性判定、自然语言指令语义拆解、模糊口语指令的标准化转换、场景动态事件描述等内容。

所有输出内容均属于认知层面的信息结果，不包含任何设备控制、运动轨迹、姿态调整相关的硬件执行参数。

- 整体运行

VLM的整体运行流程遵循标准化的跨模态拟合逻辑。

视觉端：依托视觉Transformer、DINOv2等主流视觉编码器完成图像特征提取。逐层提取画面的纹理特征、轮廓特征、局部细节特征与全局场景特征，将二维图像信息转化为高维视觉隐向量。

语言端：依托大语言模型编码器完成文本分词、语义编码与句法特征提取，将自然语言转化为语言语义隐向量。

两类异构特征输入统一的Transformer注意力主干后，通过多头自注意力机制计算视觉区域与语言词元的关联权重，完成跨模态特征的精准对齐，持续拟合图像场景与文本语义之间的对应规律，最终输出符合场景逻辑的认知结果。

VLM可直接接入VLN、VLA、世界模型等后续模块，为各类任务的推理计算提供基础环境信息。

### **VLN 视觉语言导航模型（Vision Language Navigation）**

Vision Language Navigation结合视觉观测、语言指令完成空间导航决策的功能模型。

更深层解释是指一类在视觉语言跨模态对齐基础上，叠加三维空间拓扑建模、智能体路径规划的专用具身网络模块，主要服务机器人自主移动类任务体系。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGeUbbHcEhnWGgynvZpKOejm8gDU3lu6BylwvDtWSAnZhBy1uYiaNhSAVWvxVEeJQY2DoWSEicfcx0OF9AxM8oF4Wia9w026THRIF0/640?wx_fmt=png&from=appmsg#imgIndex=7)

VLN是基于VLM跨模态感知能力迭代拓展而来的专用功能性模块，在图文语义对齐的基础上，新增三维空间几何建模与智能体运动规划能力。

- 输入

VLN的输入数据体系在VLM图文双模态的基础上完成升级拓展，除基础的图像视觉词元、自然语言指令词元外，新增三维场景深度特征、空间拓扑结构特征、障碍物坐标特征、场景连通性特征等空间维度词元。

多维度空间特征的加入，让模型可以精准感知真实物理场景的立体结构，摆脱二维平面认知的局限性，适配室内外复杂三维环境的导航需求。

- 输出

VLN的输出数据全部聚焦智能体空间移动场景，主要包含机器人底盘全局导航路径、局部实时轨迹调整参数、目标点位空间坐标、动态障碍物避让位移参数、场景切换导航序列等运动相关参数。

- 整体运行

VLN的技术运行流程采用主干复用、分支拓展的设计思路，最大程度继承VLM成熟的跨模态融合能力，保证图文语义理解的稳定性。

模型在通用图文融合主干之外，独立增设空间几何编码器与深度估计分支，专门用于解析场景深度信息、空间边界结构、障碍物分布、场景通行区域等立体空间特征。

在模型推理阶段，网络会同步融合语言导航指令、实时视觉场景、三维空间结构三类信息，通过海量真实场景导航数据的迭代训练，拟合自然语言导航需求、环境空间状态与机器人最优移动路径之间的对应关系，输出适配当前场景的导航规划结果。

- 差异化特征

VLN模型整体能力属于VLM感知能力的延伸拓展，功能边界严格限定在空间移动导航范围内。

网络内部不存在机械力控拟合、多关节协同运动、物理交互推演相关的结构分支，训练目标仅优化空间路径规划与动态避障精度，不涉及实体操作任务的损失计算。

### **VLA 视觉语言动作模型（Vision Language Action）**

Vision Language Action将视觉信息、语言指令直接映射为设备动作控制的交互模型。

专业层面指代一类端到端的具身执行架构，统一整合感知、导航、运动控制、物理交互多维度建模能力，是机器人实体作业任务的通用执行载体。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcU1pa16UTruEOvPVGHF0Ipfs2WNAhv4enu20d90ojYEtysjicIYibfgq9IX6pyc0psEWePaAvjQrej9Ff5mUUicB2bHtM3CiaIWCo/640?wx_fmt=png&from=appmsg#imgIndex=8)

VLA是整合VLM感知理解、VLN空间导航双重能力，新增全身运动与物理交互建模分支的端到端一体化交互模块，是现阶段具身智能体系中负责实体执行的核心功能载体，可完整实现智能体感知、决策、动作输出的全闭环工作流程。

- 输入

VLA的输入数据体系实现了全维度多模态融合：

整合视觉场景词元、自然语言指令词元、机器人本体姿态词元、多关节状态词元、末端执行器状态词元、环境空间特征词元六类数据，构建起覆盖环境、指令、设备状态的完整输入体系。

- 输出

VLA的输出数据为机器人硬件可直接解析执行的标准化控制参数，覆盖机器人全维度运动能力，包含底盘全局移动轨迹、局部姿态调整参数、机械臂多关节旋转角度、末端执行器开合幅度、夹持力度参数、人形机器人全身协同联动序列等精细化控制信息，可适配简单移动与复杂精细操作的各类任务需求。

- 整体运行

VLA的技术运行流程采用全端到端的一体化架构设计，摒弃传统感知、导航、动作分层独立编码、逐级传输的繁琐模式，通过单一Transformer主干网络完成所有模态数据的统一编码与特征融合。

模型将视觉像素信息、语言文本信息、机器人状态信息、空间结构信息统一转化为离散词元序列，送入主干网络完成全局注意力建模，直接拟合原始输入数据到连续动作控制序列的映射关系。

训练过程中，网络会同步优化语义对齐精度、空间定位精度、动作生成平滑度、物理交互适配性多类目标，实现多任务协同优化，大幅降低分层架构带来的特征损耗与推理延迟。

### **VLX 视觉语言通用架构模型（Vision Language X）**

Vision Language X为覆盖视觉、语言及全维度具身任务的拓展型通用架构，其中字母X为技术代称，代表未知、可拓展、全场景的任务适配属性。

整合VLM、VLN、VLA全部能力的一体化融合框架，用于适配通用机器人多任务、全场景的智能作业需求。

- 输入

VLX的输入体系完全复用前三类模型的全维度多模态输入资源，统一整合二维视觉、三维空间、自然语言、机器人本体状态、环境时序状态五类词元信息，不存在模态删减与功能拆分。

统一的输入体系让VLX能够兼容所有适配VLM、VLN、VLA的场景任务，具备极强的场景通用性与任务适配性。

- 输出

VLX的输出体系采用多分支并行推理模式，单轮网络前向推理可以同时输出三类差异化任务结果，包含：

VLM对应的场景语义描述与认知特征；

VLN对应的三维空间导航路径；

VLA对应的机器人全身动作控制指令。

实现感知、导航、执行多任务同步输出，无需多次推理、多模型串联。

- 整体运行

VLX依托统一编码、多头分支的架构设计，搭建全局共享的多模态编码器完成所有异构数据的统一特征提取，保证底层特征的一致性与互通性。

网络内部配置多组专属注意力分支与任务头，不同分支分别对应环境感知、空间导航、物理交互三类核心任务，在共享底层特征的基础上，独立完成专属任务的拟合与推理。

单次前向计算即可完成多任务并行求解，有效规避多模块串联部署带来的特征传输损耗、推理延迟、系统冗余等问题，大幅提升整体系统的运行效率。

VLX的运行适配场景具备高度灵活性，可根据实际作业需求自由切换工作模式：

> 在仅需环境监测的场景中，可单独启用感知分支完成场景认知；
>
> 在仅需自主移动的场景中，可单独启用导航分支完成路径规划；
>
> 在需要完整物理交互的复杂场景中，可全开多分支实现感知、导航、执行一体化作业，全方位适配通用机器人的多元化作业需求。

- 差异化特征

VLX的技术差异化特征主要体现在架构形态与部署模式，VLM、VLN、VLA均为单一功能导向的独立模块化模型，需要多模型串联协同才能完成完整具身任务。

VLX属于融合型顶层架构，整合三类模块的全部功能与底层资源，以单套网络结构实现全任务覆盖。

### **世界模型 WM（World Model）**

World Model聚焦时序动态建模、物理规则学习、未来场景生成的预测类网络模块，可为各类具身智能算法提供虚拟推演与状态预判的技术支撑。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/uwFbeBKoFGfOSePicbgrSJNYaHvgOGyvdSMViban4yyPbBVhGV2m6ROIEqjQ2fKxhM4rXyQbhtBbsKnNPjBxfsPILEfPABmRwEZN0qO9NRhmU/640?wx_fmt=jpeg&from=appmsg#imgIndex=9)

世界模型是依托通用神经网络拟合逻辑与多模态编码体系搭建而成。

- 输入

世界模型的输入数据由两大核心部分构成，分别是机器人实时环境观测词元、智能体候选动作序列词元。

环境观测词元包含实时视觉画面、空间结构状态、物体分布信息、场景物理参数等静态与动态环境特征；

候选动作序列词元包含VLA、VLN等模型输出的待执行移动轨迹、关节动作、姿态调整等控制序列，两类数据结合为时序推演提供完整的输入依据。

- 输出

世界模型的输出为连续时序化的环境状态信息，涵盖未来多时间帧的视觉场景画面、物体位移变化趋势、物理交互后的场景结构变化、任务演化走向、动作对应的环境反馈结果等动态信息，能够完整还原动作执行后的连锁环境变化。

下图反映端到端模型在世界模型中训练：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcRp5k7aLAmVvkSP5hlVFEDeic6k4SDS2NTBmich2JG2ALeVr2hGPXuVN1wxXDRJdIYOTyY1prWDxTXxnzGo6lW9txCrjO5GqDEg/640?wx_fmt=png&from=appmsg#imgIndex=10)

- 整体运行

世界模型的技术运行流程主要依托自回归时序建模与扩散生成建模两类主流技术方案，整体训练与推理围绕动态时序规律展开。

模型在训练阶段，会海量学习真实物理场景中的重力、摩擦力、物体碰撞、物体形变、流体运动等客观物理规律，建立标准化的物理场景认知体系；

在推理阶段，模型基于当前实时环境状态与待执行候选动作，逐帧推演未来时间维度的环境动态变化，拟合当前环境、动作输入与未来多步世界状态的动态对应关系，生成连续、符合物理规律的虚拟场景变化序列。

世界模型的运行定位区别于所有执行类模型，推理生成的虚拟场景状态、任务演化结果、环境反馈信息，不会直接下发至机器人硬件完成真实执行。

- 差异化特征

世界模型的技术差异化特征集中体现在时序建模维度与任务服务逻辑：

VLM、VLN、VLA、VLX四类模型的建模逻辑聚焦瞬时状态匹配，基于当前时刻的环境与指令，输出当下可执行的认知、导航、动作结果，服务于实时性作业需求。

世界模型的建模逻辑聚焦长时序动态演化，打破瞬时状态的局限，基于当下信息推演未来持续变化，以虚拟试错、提前预判的方式优化整体任务策略。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGe2hicpibavYlZJ0KrkB8iaiaBH735Z4H66YWqJ1UFY4TIYtibpKCPq2z9KZF4drNvkwt9unl8TBbKVfvLj69FEaD1IrUgy8k5BePib4/640?wx_fmt=png&from=appmsg#imgIndex=11)![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfwETSicDd8o3PtwqJ0ShicBC2iaYEL91iaUibPFmxSY7Uiav7Uokl1nmUEBAABqicaubX3xuxLibGEhqgDlrjrpVtwdjricjJwxSPznqFw/640?wx_fmt=png&from=appmsg#imgIndex=12)

五大模型能力层级递进与协同链路图 | 从感知、导航、执行到一体化架构，再到虚拟推演赋能。实线代表能力继承与执行链路，虚线代表虚拟预演与策略优化。©【深蓝具身智能】编译

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=13)

## **多模型同源协同的技术逻辑与关联关系**

五类模型依托统一的底层神经网络架构与多模态编码体系，整体呈现由基础认知到复杂执行、由分立模块到统一架构、由即时响应到时序预判的迭代规律。

- VLM作为整套体系的最前置模块，承担基础跨模态感知理解工作，为VLN、VLA、VLX、世界模型提供统一的环境输入素材；
- VLN在VLM二维场景认知的基础上，叠加三维空间几何建模能力，完成空间结构认知与自主导航规划，为实体交互提供空间定位支撑；
- VLA整合前两类模型的全部能力，新增多维度物理交互建模分支，构建起感知、导航、动作执行的完整闭环，实现机器人自主作业的核心功能；
- VLX对三类分立功能模块进行架构融合，以一体化设计替代分层串联模式，提升系统运行效率与通用性；
- 世界模型独立于即时执行链路，以时序推演能力赋能全链路模块，优化长时序复杂任务的整体表现。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGeDxZns2AFbbmtYvXJ6NdWEQiaYsRAwFvTrR682dL6VTlW7UXCB3tlFcMzSKxacCveABQjQkrWS0yWmnynV8F55bmvZKoIhiayk8/640?wx_fmt=png&from=appmsg#imgIndex=14)

VLN与VLA具备高度同源的技术底座与能力包含关系，两类模型共享视觉编码器、语言编码器与基础跨模态融合主干，仅在任务输出分支与能力覆盖范围存在差异。

VLN属于单一功能模块，整体架构围绕空间移动导航搭建，仅配置底盘运动轨迹与避障参数输出分支，不具备实体交互操作能力，适用于无接触、纯移动的简单场景。

VLA属于全功能执行模块，网络内部完整搭载VLN的空间导航建模分支，可独立完成所有VLN适配的导航任务，同时拓展机械臂操作、多关节联动、力控调节、全身协同运动等进阶能力。

VLA与世界模型的多模态编码规则、特征表征维度、物理规律学习体系完全互通，数据与模型可以无缝对接协同，是目前行业主流的具身智能组合范式。

- VLA聚焦瞬时状态的即时决策，基于当前环境观测与任务指令，生成多组可落地的机器人动作方案，解决作业过程中当下的动作选择与执行问题；
- 世界模型聚焦长时序状态的未来推演，接收VLA输出的候选动作序列，在虚拟仿真空间中逐帧推演每一组动作对应的环境变化、任务走向与潜在风险，完成多方案的对比筛选与优化迭代。

二者搭配运行，可以构建出“即时决策+虚拟预演+择优执行”的进阶作业链路，有效降低真实场景试错成本，提升复杂动态场景、长流程任务的完成质量。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=15)

## **五类模型：同源、异构、协同**

VLM、VLN、VLA、VLX、世界模型五类具身智能模块，整体技术体系扎根于统一的神经网络数据拟合原理与多模态混合编码机制，底层网络架构、特征提取逻辑、跨模态融合方式、物理规律学习体系保持高度一致。

当下具身智能技术仍处于高速迭代阶段，分立模型高精度落地、一体化通用模型迭代、世界模型赋能优化，成为未来产业发展的三大核心趋势。

技术演进的加速度往往超出预期。

当我们今天还在逐一拆解VLM、VLN、VLA、VLX和世界模型时，明天的行业头条或许就会出现一个重新定义一切的通用具身大模型。

而理解它们的底层逻辑，正是提前看懂那个未来的入场券……

编辑｜咖啡鱼

审编｜具身君

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=16)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGehZibr7tZF2zldbePrhVEqzN7MibldHydGKe6nGybQEX1BRAILTtBAjAjRnXgTvkfibaHsrYOzt70Uiaiclh9cuFVkcIMdff3SoyXs/640?wx_fmt=png&from=appmsg#imgIndex=17)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=18)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=19)

点击❤收藏并推荐本文**
