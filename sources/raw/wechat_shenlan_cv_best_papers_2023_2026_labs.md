# 四年十篇顶会Best Paper，谁在持续霸榜？

> 原始抓取（2026-11-01，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/CJki5-KhJ438RdAIhzNbCw

四年十篇顶会Best Paper，谁在持续霸榜？
深蓝学院 深蓝具身智能
 Nov 1, 2026, 9:07 AM

「盘点CV风向标」

在计算机视觉领域，CVPR、ICCV和ECCV被并称为“三大会”，代表着该领域学术研究的最高水平。每年评选出的Best Paper，不仅是当年最具突破性的技术成果，更是未来几年AI行业的发展风向标。

近四年来，我们见证了CV技术从基础的感知、理解，大步迈向生成式AI、端到端自动驾驶，乃至更复杂的物理世界交互。

这些突破性成果的背后，站着一群全球顶尖的AI实验室。

本文，我们以时间为线索，盘点近四年CV三大会的Best Paper，看看这些顶尖工作出自哪些实验室。

注：本文所涉及的各实验室背景、人员归属及研究方向，均基于公开可获取的论文、机构官网及学者个人主页等资料整理而成，信息截至本文发布时间。由于学术界人员流动频繁，实验室方向也在持续演进，相关信息未必能做到实时更新，如有疏漏或出入，欢迎读者在评论区留言补充或指正，我们将及时核实修订。

欢迎关注【深蓝AI】 
将持续分享人工智能领域前沿动态👇




深蓝AI




1
—
2023年
2023年是生成式AI与大模型在视觉领域大放异彩的一年，CVPR和ICCV的Best Paper精准地捕捉到了这一趋势。


1. 视觉编程的先驱：VisProg

会议： CVPR 2023 Best Paper

论文： Visual Programming: Compositional Visual Reasoning Without Training

实验室： 艾伦人工智能研究所（AI2）PRIOR团队

技术看点：长期以来，处理复杂的视觉任务往往需要针对每个特定任务训练专门的模型。而VisProg提出了一种"神经符号"系统，利用大语言模型的上下文学习能力，将自然语言指令转化为一段Python代码（视觉程序）。这段程序可以自动调用各种现成的计算机视觉模型（如目标检测、图像分割、VQA模型等）来逐步解决复杂的视觉推理任务，且完全不需要额外的训练。

图1 | VisProg根据自然语言指令生成视觉程序，并逐步调用专门模块完成图像编辑和推理任务©【深蓝 AI】编译

实验室介绍：艾伦人工智能研究所（Allen Institute for AI，简称AI2）由已故微软联合创始人保罗·艾伦（Paul Allen）于2014年创立，是一家总部位于西雅图的非营利性研究机构，使命是"为了公共利益推进AI发展"。

VisProg的两位作者Tanmay Gupta和Aniruddha Kembhavi均来自AI2的PRIOR（Perceptual Reasoning and Interaction Research）团队。该团队长期致力于计算机视觉、具身智能和机器人技术的交叉研究，以开放的研究文化和大量开源工具著称。

VisProg正是他们将大模型推理能力与传统CV模块完美结合的典范之作——它无需任何额外训练，仅凭"写程序"就能解决组合式视觉推理，这一思路在当时极具前瞻性。



2. 端到端自动驾驶的里程碑：UniAD

会议： CVPR 2023 Best Paper

论文： Planning-oriented Autonomous Driving

实验室： 上海人工智能实验室 OpenDriveLab

技术看点：在自动驾驶领域，传统的系统通常被拆分为感知、预测、规划等多个独立模块，这容易导致误差累积。UniAD（Unified Autonomous Driving）打破了这一常规，提出了首个以规划为导向的端到端自动驾驶网络框架。它将所有驾驶任务整合到一个统一的网络中，通过查询（Query）机制连接各个节点，实现了从传感器输入到最终驾驶动作规划的联合优化，大幅提升了系统的安全性和鲁棒性。

图2 | UniAD的端到端框架，将感知、预测和规划任务整合在一个统一的网络中，以规划目标驱动全局优化©【深蓝 AI】编译

实验室介绍：上海人工智能实验室是我国人工智能领域的新型研发机构，成立于2020年。其中，OpenDriveLab由李弘扬博士（现任香港大学助理教授）领导，是全球端到端自动驾驶研究的领军团队之一。

UniAD的第一作者胡一鸣（Yiming Hu）等人在李弘扬的带领下，提出了这一具有里程碑意义的框架。该团队坚持"以终为始"的研究理念，推动了自动驾驶技术从模块化向端到端大模型的范式转变，并持续开源了多个极具影响力的自动驾驶数据集和评测基准。



3. AIGC时代的控制神笔：ControlNet

会议： ICCV 2023 Best Paper（Marr Prize）

论文： Adding Conditional Control to Text-to-Image Diffusion Models

实验室： 斯坦福大学 Brown媒体创新研究所

技术看点：虽然Stable Diffusion等文生图模型能生成高质量图像，但如何精确控制生成内容的姿态、边缘和结构一直是个难题。ControlNet巧妙地解决了这一痛点，它通过锁定大型预训练扩散模型的原始权重，并克隆一个可训练的副本，将Canny边缘、人体姿态（OpenPose）、深度图等额外条件注入到生成过程中。这使得AI绘画从"开盲盒"变成了可精确操控的创作工具。

图3 | ControlNet通过锁定原始权重并训练零卷积（Zero Convolution）层，实现对扩散模型的精确空间控制©【深蓝 AI】编译

实验室介绍：ControlNet的诞生充满了传奇色彩。它的第一作者张吕敏（Lvmin Zhang）是斯坦福大学计算机科学系的博士生，导师是Maneesh Agrawala教授——斯坦福Brown媒体创新研究所所长、Forest Baskett计算机科学讲席教授。

Agrawala教授长期致力于设计能增强人类理解和创作能力的视觉接口，其实验室在可视化与创意工具领域积累深厚。ControlNet在GitHub上开源后迅速爆火，斩获数万Star，彻底改变了整个AI绘画社区的生态，成为了Stable Diffusion时代最不可或缺的插件之一。

值得一提的是，ControlNet与同年ICCV的另一篇Marr Prize得主——多伦多大学DGP实验室的"被动超宽带单光子成像"共同分享了这一殊荣，两篇论文分别代表了AI生成与物理成像两个截然不同的研究方向。





深蓝AI




2
—
2024年

2024年，CVPR和ECCV的Best Paper呈现出截然不同的两种风格：前者是Google Research在生成式AI领域的强势收割，后者则是对极简主义硬件设计的深刻思考。



1. 让静图"活"起来：Generative Image Dynamics

会议： CVPR 2024 Best Paper

论文： Generative Image Dynamics

实验室： Google Research

技术看点：自然界中的运动——树叶的摇曳、花朵的颤动、蜡烛的跳动——遵循着特定的物理规律，这些运动往往是振荡性的、可预测的。Generative Image Dynamics提出了一种对图像空间场景运动建模的生成先验，通过在频域（傅里叶域）中学习运动的"频谱体积"（Spectral Volume）表示，能够从单张静态图像中生成无缝循环的动态视频，或允许用户通过拖拽与图像中的物体进行交互式动态模拟。

图4 | 从单张输入图像出发，模型预测频谱体积（图像空间模态基），并将其转化为循环视频或交互式动态效果©【深蓝 AI】编译

实验室介绍：这篇论文的四位作者——Zhengqi Li、Richard Tucker、Noah Snavely、Aleksander Holynski——全部来自Google Research（现Google DeepMind）。

其中，Noah Snavely是计算机视觉领域的知名学者，曾任康奈尔大学教授，以Photo Tourism和NeRF相关工作著称，后加入Google Research。Aleksander Holynski同样是图像动画领域的重要研究者。这支团队深耕视频生成与3D视觉的交叉地带，Generative Image Dynamics的成功，是Google Research在生成式视觉领域长期积累的集中体现。



2. 让AI理解图像质量的细粒度反馈：Rich Human Feedback

会议： CVPR 2024 Best Paper

论文： Rich Human Feedback for Text-to-Image Generation

实验室： Google Research（联合UCSD等机构）

技术看点：现有的文生图评测指标（如FID、CLIP Score）往往只给出一个单一的数值评分，无法定位图像中具体的问题区域。这篇论文提出了"丰富人类反馈"（Rich Human Feedback）的概念，收集了18K张生成图像的精细化标注数据集（RichHF-18K），包括：标注不合理区域的热力图、标注文本-图像不匹配的关键词，以及四个维度（合理性、对齐度、美学、整体质量）的细粒度评分。在此基础上，他们训练了一个多模态Transformer模型（RAHF），能够自动预测这些丰富的反馈信号，并用于改进文生图模型的训练数据筛选和图像修复。

图5 | RAHF模型包含视觉流和文本流两个计算分支，通过自注意力融合后，分别预测不合理区域热力图、评分和误对齐关键词序列©【深蓝 AI】编译

实验室介绍：这篇论文的通讯作者Junfeng He来自Google Research，第一作者Youwei Liang当时是加州大学圣地亚哥分校（UCSD）的博士生，在Google实习期间完成了这项工作。整个团队横跨Google Research、UCSD、南加州大学、剑桥大学等多个机构，体现了工业界与学术界的深度协作。

2024年CVPR的两篇Best Paper均出自Google Research，这一事实充分说明了Google在生成式视觉AI领域的统治性地位——无论是让静图动起来，还是让AI学会"挑剔"地评价图像，Google都走在了最前沿。



3. 探索视觉的最小极限：Minimalist Vision

会议： ECCV 2024 Best Paper

论文： Minimalist Vision with Freeform Pixels

实验室： 哥伦比亚大学CAVE实验室

技术看点：传统的摄像头通常捕捉数百万像素的图像，这不仅消耗大量电能，还容易引发隐私问题。这篇论文提出了"极简视觉"（Minimalist Vision）的概念，探讨了一个有趣的问题：为了完成特定任务（如统计房间人数或检测门是否打开），最少需要多少个像素？他们设计了一种带有"自由形态像素"（Freeform Pixels）的原型相机，仅用极少数（如几个或十几个）精心设计的、形状不规则的像素，就能在极低功耗下完成任务，同时从物理层面上保护了隐私。

图6 | 极简视觉系统通过少量自由形态像素实现房间人数统计，既大幅降低了功耗又从硬件层面保护了隐私©【深蓝 AI】编译

实验室介绍：这篇论文出自哥伦比亚大学的计算机视觉实验室（CAVE Lab），由著名的Shree K. Nayar教授领导。Nayar教授是计算成像（Computational Imaging）领域的先驱，曾于1990年和1995年两度荣获ICCV的David Marr Prize，是极少数能多次斩获这一殊荣的学者之一。

CAVE实验室的风格独树一帜，他们不盲目追逐大模型，而是深耕物理学、光学与计算机视觉的交叉领域。这篇论文由Nayar教授和他的博士生Jeremy Klotz合作完成，展现了他们在硬件传感器设计上的深厚底蕴。在算力狂飙的时代，这种"少即是多"的极简哲学，反而显得尤为珍贵。





深蓝AI




3
—
2025年

进入2025年，CV研究的重心明显向三维世界和物理规律的理解倾斜，这与具身智能的崛起密不可分。



1. 3D重建的统一架构：VGGT

会议： CVPR 2025 Best Paper

论文： VGGT: Visual Geometry Grounded Transformer

实验室： 牛津大学VGG组 + Meta AI

技术看点：传统的三维重建（如SfM）通常需要复杂的流水线和耗时的后处理优化。VGGT（Visual Geometry Grounded Transformer）提出了一种前馈神经网络，能够在单次前向传播中，直接从单张或多张图像中推断出场景的所有关键3D属性，包括相机参数、点云图、深度图和3D点轨迹。它不仅速度极快（秒级输出），而且在许多场景下超越了需要复杂优化的传统方法，真正实现了"一网通吃"的3D理解。

图7 | VGGT将输入图像切片后，通过交替的帧内和全局自注意力层，直接预测所有密集的3D属性，包括相机参数、点云和深度图©【深蓝 AI】编译

实验室介绍：提到计算机视觉，就绕不开牛津大学的视觉几何组（Visual Geometry Group，简称VGG）。这个由Andrew Zisserman教授等人创立的实验室，曾在深度学习早期提出了著名的VGGNet，深刻影响了CNN的发展，在多视图几何领域更是积累了数十年的深厚底蕴。

VGGT的第一作者Jianyuan Wang是牛津大学VGG组与Meta AI（FAIR）的联合培养博士生，导师包括Andrea Vedaldi、Christian Rupprecht（牛津）和David Novotny（Meta AI）。这项工作不仅延续了VGG组在几何视觉领域的传统优势，更结合了Meta AI在大规模Transformer工程上的强大能力，代码开源在facebookresearch/vggt。



2. 让AI理解物理搭建：BrickGPT

会议： ICCV 2025 Best Paper（Marr Prize）

论文： Generating Physically Stable and Buildable Brick Structures from Text

实验室： 卡内基梅隆大学（CMU）计算机科学系

技术看点：生成3D模型已经不新鲜，但生成的模型是否符合物理规律、能否在现实中搭建出来？BrickGPT首次解决了这个问题。它是一个能够根据文本提示，生成物理上稳定、且可由真实玩具积木拼装的三维结构模型。研究团队构建了一个大规模的物理稳定积木数据集（StableText2Brick，包含超过47,000个积木结构），并微调了自回归大语言模型，使其不仅能理解几何形状，还能在生成过程中进行物理稳定性检查和回滚（Physics-aware Rollback），确保每一块积木都满足装配约束。生成的结构不仅可以由人工手动拼装，还可以由机械臂自动完成组装。

图8 | BrickGPT的训练与推理流程，模型在自回归生成积木结构时会进行严格的物理稳定性检查与回滚©【深蓝 AI】编译

实验室介绍：这篇斩获Marr Prize的论文来自卡内基梅隆大学（CMU）计算机科学系的Generative Intelligence Lab，由Jun-Yan Zhu教授领导。Jun-Yan Zhu是GAN领域的知名学者，以CycleGAN、pix2pix等工作享誉学界。

第一作者Ava Pun是该实验室的博士生，论文同时得到了CMU机器人研究所Changliu Liu教授（iControl Lab）和Deva Ramanan教授的联合指导，体现了计算机图形学与机器人学的深度交叉。

BrickGPT的成功，标志着AI生成内容（AIGC）正在从纯粹的视觉像素生成，走向符合物理现实约束的实体生成，这对于具身智能和机器人仿真具有重大意义。





深蓝AI




4
—
2026年

随着技术的演进，计算机视觉的终极目标之一——理解动态的三维世界（即4D），在2026年迎来了重大突破。



1. 高效的4D动态重建：D4RT

会议： CVPR 2026 Best Paper

论文： Efficiently Reconstructing Dynamic Scenes One D4RT at a Time

实验室： Google DeepMind + 牛津大学 + 伦敦大学学院（UCL）

技术看点：理解视频中动态场景的复杂几何和运动轨迹，一直是CV领域的巨大挑战。D4RT（Dynamic 4D Reconstruction and Tracking）提出了一种统一的Transformer架构，从单段视频中联合推断深度、时空对应关系和相机参数。它的核心创新在于一种新颖的查询机制，允许模型独立、灵活地探测时空中任意一点的3D位置，避免了密集的逐帧解码。这使得D4RT在动态场景重建和像素级跟踪任务上，达到了前所未有的效率和精度，在所有审稿人评分中获得了满分（straight 6s）。

图9 | D4RT通过统一的Transformer架构，高效地重建视频中动态场景的3D运动轨迹、深度图和相机参数©【深蓝 AI】编译

实验室介绍：D4RT的第一作者Chuhan Zhang（张楚晗）在牛津大学VGG组完成博士学位（导师同为Andrew Zisserman教授），毕业后加入Google DeepMind担任高级研究科学家。这项工作正是他在Google DeepMind期间，联合Guillaume Le Moing、Skanda Koppula、Ignacio Rocco等同事，以及来自UCL和牛津大学的合作者共同完成的。

值得注意的是，CVPR 2025的Best Paper VGGT（牛津VGG组）和CVPR 2026的Best Paper D4RT（Google DeepMind，第一作者来自牛津VGG组）之间存在明显的传承关系——D4RT可以视为将VGGT的静态3D重建范式，扩展到动态4D时空的直接延伸。让AI像人类一样理解运动中的3D世界，不仅是计算机视觉的突破，更是构建通用世界模型的关键基石。





深蓝AI




5
—
技术河流的分与合

纵观近四年的CV三大会Best Paper，可以清晰地看到一条技术演进的主线：

计算机视觉正在从静态的2D像素处理，全面走向动态的、符合物理规律的3D/4D世界重构。

通过统计各顶尖实验室在近四年中斩获的Best Paper数量（包括并列情况），我们可以直观地看出当前学术界和工业界的研究热点和主导力量：

从依托大模型做视觉推理、精准可控AI生成，再到端到端自动驾驶落地、极简视觉硬件创新，直至如今兼顾物理约束的实体生成、动态4D场景重建，顶会最佳论文始终精准预判行业风向。

同时我们也能清晰看见学界与工业界的格局变化：谷歌持续领跑生成式视觉与4D重建，牛津VGG组坚守几何视觉底盘稳居学术第一梯队，顶尖成果无一不是算法、硬件、物理学科与大模型的跨界融合。

未来，随着具身智能、世界模型持续爆发，计算机视觉也将继续迎来颠覆性突破。



推荐阅读



【深蓝具身智能】的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️




点击❤收藏并推荐本文

​
深蓝具身智能
Like
Share
Boost
Comment