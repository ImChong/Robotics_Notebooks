---
title: 自动驾驶感知算法盘点｜目标检测篇（一）
author: 深蓝AI
date: "2026-06-14 12:09:00"
source: "https://mp.weixin.qq.com/s/7Mm5OwVKgoyT4Zpr45E34A"
---

# 自动驾驶感知算法盘点｜目标检测篇（一）

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAKPXr0kicZddyXdPOg1Jm7tKusPLcWicG0ALpMqpjSHZxxsu45C13rzA4XZ2leKiaxG64fPqc9zIRIj8CR43YYibVy9ic8aRib3LUd8/640?wx_fmt=png&from=appmsg#imgIndex=0)
> 「深蓝学院」推出《自动驾驶感知算法盘点》系列专栏。我们将逐一拆解感知领域的关键技术方向，为大家建立完整清晰的技术图谱。
>
> 本文是该系列的第一篇：目标检测。

---

自动驾驶感知系统是智能汽车实现自主决策、安全行驶的核心基础，相当于车辆的“眼睛”和“视觉大脑”。作为专栏的开篇之作，我们聚焦于感知系统最基础、也是最关键的任务之一：目标检测。目标检测作为感知模块的核心任务，主要分为2D目标检测与3D目标检测两大技术体系。（限于篇幅原因，本篇先盘点2D目标检测）

我们按技术架构将算法划分为**两阶段Anchor-Based算法、单阶段CNN Anchor-Based算法、单阶段Anchor-Free算法、Transformer视觉骨干检测算法**四大类，贴合自动驾驶车载RGB图像检测场景。

每个算法我们都尽可能直白讲解原理、指明核心创新，提炼一下共同点（都是同一个大思想下的小改动），顺便也说说车载落地适配性，贴合自动驾驶车辆、行人、交通标识、遮挡小目标检测需求。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpBicBm6Y6L98w85KjaHrRID3Ricb5nvdyptKBp1MupS3caYc5gT3rurQqicjPa0qOzJ4UhVsb1FGVHOM1080xtIpQ5NYqmNtmJP6I/640?wx_fmt=png&from=appmsg#imgIndex=1)**关注并私信0614**免费获取目标检测算法汇总包👇

***深蓝AI***

**1****—******经典算法****盘点****![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpBwBJnhMPDUOVGolOComxEOQxo2Db1f1wkX8rbXcP3zfiao5owENGHu9t2nmvMBfT28ncY89iakafHEb05qFuPmNmjmDicJhmoa5M/640?wx_fmt=png&from=appmsg#imgIndex=2)
**两阶段Anchor-Based经典算法**

它们的共性是遵循「候选框生成→特征精修分类回归」双流程，先筛前景候选区域，再微调边框、完成分类，误检率低、密集遮挡目标精度高；缺陷是推理链路长、帧率低，车载边缘算力部署压力大，均依托锚框Anchor设计。

（1）R-CNN

- 简介：选择性搜索提取2000个图像候选框，逐个裁剪送入CNN提特征，搭配SVM完成分类、边框微调。
- 亮点：首次将卷积深度学习引入自动驾驶2D检测，替代传统HOG人工特征，检测精度跨越式提升。
- 局限：但是候选框重复卷积、速度极慢，无法车载实时部署。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpCFKD92bibFaBOX3IXfciaV4Wd9u41GjzRAz2MspPc6PftCIRstHB73ljic0ygr0qic6axZ8WN7KC1glZ4E2k7mx1UhxeNTWPtaHTE/640?wx_fmt=png&from=appmsg#imgIndex=3)

（2）Fast R-CNN

- 简介：对整张图像仅卷积提取一次全局特征，通过RoI池化层截取候选框特征，分类、边框回归联合训练。
- 亮点：共享图像卷积特征、取消重复计算，合并训练流程，相较R-CNN提速多倍；解决模型分段训练、特征冗余痛点。
- 局限：保留选择性搜索候选框，无网络自主生成候选区，速度大幅优化但未摆脱传统区域搜索。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpBibqlwZu8f5AqA7kIpqDEnXRWQDQQ7Z4CfK4oKDHMqyMibDfQqQtx6TY4xeHPcLt5sibNMia05j27PuXKlXDnYbNhGULxBLbhg3c4/640?wx_fmt=jpeg#imgIndex=4)

（3）Faster R-CNN

- 简介：原理是骨干网络提特征，新增RPN区域建议网络自主生成Anchor锚框与候选区，RoI池化后完成检测，完整端到端卷积架构。
- 亮点：用神经网络RPN替代选择性搜索，实现首个全深度学习两阶段检测器，成为自动驾驶高精度检测基线。

  是同系列精度最优、推理最快，是R-CNN、Fast R-CNN终极优化版本，量产高精度感知基线模型。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpApl3tq4VkSu6K9ibzzlPns7EDUwHQm2pv1ic5Gj4d5JvEmXsOtd2rnAMMKqCSQdKwJjORiaEVhMemeVLRckgkEDPjdneqKtSUWms/640?wx_fmt=png&from=appmsg#imgIndex=5)**后台私信0614，免费获取目标检测算法汇总包**
**单阶段CNN Anchor-Based算法**

共同特点是舍弃独立候选框分支，整张图像单次前向推理，同步输出类别+边框，推理速度远快于两阶段模型；依托预设Anchor，车载帧率高、轻量化易部署，但是小目标、正负样本失衡为通用短板。

包含SSD、RetinaNet、EfficientDet、YOLOv3/v4/v5/v7/v8/v9全系列。

（1）SSD

- 简介：是多尺度特征金字塔分层检测，浅层特征查远距离交通小目标，深层特征查近处车辆，多尺寸Anchor匹配目标。
- 核心创新：首创单阶段多尺度检测架构，兼顾YOLO速度与Faster R-CNN定位精度，适配车载大小交通目标。
- 异同：比初代YOLO小目标精度更强，无复杂骨干、部署简单，正负样本不均衡缺陷未解决。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpCBiaO16ozicVBoYH6bciaSqjmHrx9Dv9ldcKdfz9gfnpSJZj3gLSBZ9D7kXNX7GyOYoIpVQSQZKaPQxwxEXBLE78SVNJ0z1XvnNA/640?wx_fmt=png&from=appmsg#imgIndex=6)

（2）RetinaNet

出自何恺明团队的经典算法

- 简介：沿用FPN特征金字塔+SSD锚框架构，更换损失函数优化模型训练。
- 核心创新：提出Focal Loss焦点损失，压制背景负样本权重、解决单阶段正负样本失衡痛点，单阶段精度追平两阶段算法。
- 效果：架构贴合SSD，核心革新损失函数，拥堵车流、复杂背景自动驾驶检测效果大幅提升。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpCfHyzsp7qqIzWxtgx4YrRVu7zhhBKU62EgGaQicEGp181UANMD3PPRbu5fajYvodB7s0C2Mcbib77WPqOyTgZrJSX1b4FQj2TuA/640?wx_fmt=png&from=appmsg#imgIndex=7)

（3）EfficientDet

来自Google Brain团队，EfficientDet是一系列可扩展的高效的目标检测器的统称。

- 简介：复合缩放统一优化骨干网络、特征金字塔、检测头维度，双向加权FPN融合多尺度特征。
- 亮点：提出模型复合缩放策略、BiFPN双向特征融合模块，参数更少、多尺度目标泛化性更强，雨雪天气图像特征融合效果优异。
- 异同：相较SSD、RetinaNet特征融合更高效，模型轻量化、泛化性拉满，车载算力适配性更强。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpBEpKT95jAvBAKTicShupnpInpcAXiblXxansMy34jcxiamWhCUoDBRAsQBaQytqQ83ic3QqNfJSkucLUWzm6vvPjk8KNic5cy3mb7M/640?wx_fmt=png&from=appmsg#imgIndex=8)

（4）YOLOv3

- 简介：引入Darknet-53骨干、三尺度FPN检测，网格预测Anchor，端到端回归检测。
- 亮点：主要是替换深层骨干、多尺度小目标优化，平衡自动驾驶检测速度与精度，工程落地性大幅改善。

初代均衡型YOLO，弥补v1/v2小目标漏检缺陷，算力开销适中。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpDMwes16YnumOictpRc2bJShh3po4D8LibU02yJTKgmywUtHWQLbm2r0Z2T0ASVtrbN9CZyPIgjJhrreDkicQc9IhGViatBT81meT4/640?wx_fmt=png&from=appmsg#imgIndex=9)

（5）YOLOv4

- 简介：优化骨干激活函数、Mosaic车载图像增强、CIoU边框损失。
- 核心创新：新增自动驾驶路况专属数据增强、优化损失函数，逆光、夜间路况鲁棒性提升。
- 异同：纯工程trick优化，网络骨架不变，恶劣路况适应性优于v3。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpB5viaYqKAy1szbKic758UINXdzic3Kxjx0pMibUqxEqzExZhEibIksPwYDQmuKqVxpPSM6w7Gic40NE2kjcS52PDzOL5ciaresE1hzY0/640?wx_fmt=png&from=appmsg#imgIndex=10)

（6）YOLOv5

- 简介：轻量化C3骨干、自适应锚框、自适应图像缩放，模型尺寸压缩。
- 亮点：实现极致轻量化、自动适配车载图像输入尺寸，嵌入式车载芯片快速部署。

参数量远低于v4，推理帧率更高，低成本量产车型首选。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpAFEzrFRCZniafAicISNXH72867TdqqibUlg2v9bhb8rIjjbtv3oic9SbEdBFkrticSEEHiaa6Gyxl20yXncNMHHAUGL00iaBRuzRe35k/640?wx_fmt=png&from=appmsg#imgIndex=11)

（7）YOLOv7

- 简介：ELAN高效聚合模块、辅助检测头协同训练。
- 亮点：高效梯度分流结构，不涨算力前提下提升精度，高速自动驾驶实时检测标杆。

速度、精度双向超越v5，中高算力车载平台主流选型。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpA7g9xb6icD46RfbfSmJbe3PsURF5S7251ibBhnmbcGqvLiavqLKKKbicXFBgg8rEMM9QGiaO1yycoKemkImib078zrqDsPpicIKoNljc/640?wx_fmt=png&from=appmsg#imgIndex=12)

（8）YOLOv8

- 简介：解耦检测头、骨干结构重构，检测+分割双分支联动。
- 核心创新：检测、实例分割一体化，适配拥堵路段目标轮廓分割，兼顾感知与后处理规划。

新增分割分支，密集车流场景优于v7，算法功能多元化。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpC2CyI5P6TID0USiaOtLNSmMD2eblhz4KK6Eia17MianWx0jP6uRiaNH4PGUYf8w7fy1IVEFLOqFLrJlXm1rYgy1YX7DAEzrRXlp8I/640?wx_fmt=png&from=appmsg#imgIndex=13)

（9）YOLOv9

- 简介：引用GELAN结构、可编程信息蒸馏、隐式特征复用。
- 亮点：主要是解决CNN深层特征退化问题，道路长尾障碍物、异形目标特征留存能力更强，2024卷积YOLO收官版本。

ps：因为我们是讨论经典算法，所以v10/v11...版本等待时间证明。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpAEJZw8HCC4mF9tyBwibaxfM8JIKoomia5xbpaCwaPerWuYDKEn3FlePfR4ibCicZ9H0kwXe5TAHkgicF7n7blDZDrPdXZcyq03U5iao/640?wx_fmt=png&from=appmsg#imgIndex=14)**后台私信0614，免费获取目标检测算法汇总包**
**单阶段Anchor-Free算法（无锚框、低参数量、轻量化）**

摒弃人工预设Anchor锚框，消除锚框超参调试成本、边框冗余计算，结构极简、训练更简单；分为中心点预测、角点预测两类，自动驾驶形变目标适配性更强。包含CenterNet、CornerNet。

（1）CenterNet

- 简介：不预设锚框，预测目标物体中心点+宽高尺寸，以中心点替代边框完成检测。是Anchor-Free主流标杆，去除NMS后处理、简化推理流程，模型延迟更低。
- 亮点：相较锚框模型，无锚框调试成本，车载推理延迟更低，中等尺寸车辆、行人检测最优。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpD9iaOjp41y0KjdndE9tlV87DtWfkH9uJQnPOtcFPB6icrPoguYMMSGiaBvsLHmjzZSzDjFR9xwbR8lFYIhvPU9jQq5h4CufsMAfs/640?wx_fmt=png&from=appmsg#imgIndex=15)

（2）CornerNet

- 简介：预测目标左上角、右下角一对角点，匹配角点组合生成检测框。首创角点检测范式，脱离中心点、锚框双重约束，异形障碍物、极端形变目标适配性更强。
- 异同：比CenterNet适配不规则道路障碍物，但角点匹配易出错，常规交通目标精度弱于CenterNet。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpCOuN1uNuwzbO1MgAV2Elfj9P0w0WUiauR1Es3icN8KpW8o8fia332be7sNpmuxBygMqialgqxQte9EP5fS210lp6NM50lSMuC29pk/640?wx_fmt=png&from=appmsg#imgIndex=16)
**Transformer骨干检测算法**

### **这一类算法依托自注意力机制建模图像全局像素依赖，摆脱**CNN局部卷积视野局限，车流重叠、遮挡、远距离目标精度拉满；分为纯视觉Transformer骨干、DETR检测架构两类，高阶自动驾驶视觉基线。

（1）ViT

- 简介：图像切块序列化输入，多头自注意力全局提取图像语义特征，替代CNN卷积骨干。
- 亮点：视觉任务首个纯Transformer骨干，全局语义建模，适配大场景道路全局感知。彻底舍弃卷积，全局视野最优，车载小数据集训练收敛慢。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAgvk8fD1508BPVp6lHhKFNjLnUSxBeFjbesmgouWmHTbschoDn31HVjsLkgib61KqWf6GSlVOAcialsibQ5D15uRiaKiaHgV8wb3CI/640?wx_fmt=png&from=appmsg#imgIndex=17)

（2）BEiT

- 简介：掩码图像自监督预训练Transformer骨干，修复图像掩码特征，强化纹理提取。
- 亮点：图像掩码预训练，车载暗光、雾化破损图像特征修复能力极强，恶劣天气适配ViT。属于是基于ViT优化自监督预训练，自动驾驶失真图像鲁棒性远超原生ViT。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpBzVsoqVOXsoicJuCDdzoaA6WxQptcg9EpwR6s6o3xyicn7n6rZiaocELUP3MiapdtWfpeRhnoybdPR4fVB8XvvZGzjIjYuJt5kbicA/640?wx_fmt=png&from=appmsg#imgIndex=18)

（3）DeiT

- 简介：通过知识蒸馏轻量化Transformer，依托CNN模型蒸馏压缩Transformer体量。
- 亮点：轻量化蒸馏Transformer，大幅降低算力开销，适配车载边缘芯片部署。精度逼近ViT，参数量、推理速度全面优化，落地性优于ViT、BEiT。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpDdbjQ3aBjhAB8In04cibpFgq0WTlmL42c5GMg8wmOqicBZuQFtB8PB4Tz5ylCiaO2dSqZbdRyGb1Eiam814za8G3lovSiceOg5RS0A/640?wx_fmt=png&from=appmsg#imgIndex=19)

（4）Swin Transformer

- 简介：是滑动窗口分层注意力、层级化特征提取，窗口内注意力降低算力。窗口注意力+金字塔层级结构，兼顾全局建模与轻量化，成为自动驾驶最优Transformer骨干。
- 亮点：相较ViT算力需求暴跌、收敛更快，兼顾精度与速度，工业界感知首选Transformer骨干。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpC3a2ofpmlvEZmiaZic8NNZSTeMpQv8z7cEa5IUKyKqCxct41qMw2a7nFVPdgRZKKFX7UiblStNn1icBEzKfOp8ItvurarK1aR9bm8/640?wx_fmt=png&from=appmsg#imgIndex=20)

（5）PVT

- 简介：是金字塔分层Transformer、渐进式下采样，轻量化多级语义特征。
- 核心创新：纯Transformer轻量化金字塔结构，替代CNN-FPN，参数量低于Swin。
- 亮点：比Swin更轻量化、车载帧率更高，精度小幅下降，低成本视觉模型优选。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpDTJBDPnV2cAhQ6cltQfIJKKa3REcx6A8zicm7LghxltyZQjtkDcmAzjXaGAKtbSE1iabZld2LYCibeIQ6f1e4U8FGmibWbAiaznibQw/640?wx_fmt=png&from=appmsg#imgIndex=21)

（6）DETR

- 简介：Transformer编码器+解码器，端到端直接输出检测结果，舍弃Anchor、NMS后处理。
- 亮点：属于首个Transformer端到端2D检测器，简化检测流水线，重叠目标误检率大幅降低。
- 局限：CNN检测器后处理全部取消，重叠车流检测效果比较好，但收敛慢、小目标精度差。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpCiciaHQcvY4t1icyATakFuU3EOTCy9DdCTAickOBpBm4P0meAjtR9gMSic68o7T5buLGxuAGt2l6ArdhkzuBJib592pTqhKIAjZSzas/640?wx_fmt=png&from=appmsg#imgIndex=22)

（7）Deformable DETR

简介：采用可变形稀疏注意力，仅聚焦目标像素做注意力计算。依靠稀疏注意力提速、优化小目标特征，修复DETR收敛慢、远距离路标漏检缺DETR量产优化版本，适配自动驾驶远距离、小尺寸交通标识检测，车规Transformer检测基线。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpB8oic1KIeV4ZlZOK0prUicgedl5RibHYa2oDwTBDwKP6TqDmCroLjsAicNv4vUZtZibIuAXgF2txPwz6XDHp19Hq9vyFXxHBwibnUgs/640?wx_fmt=png&from=appmsg#imgIndex=23)**后台私信0614，免费获取目标检测算法汇总包****2****—****技术挑战及未来方向**

自动驾驶2D目标检测从CNN卷积两阶段、单阶段经典架构，迭代至Transformer端到端检测体系，YOLO、Faster R-CNN、DETR 三大系列支撑了近十年辅助驾驶的量产落地。

相较于复杂多模态3D感知，优化后的新一代2D检测算法成本优势显著，仍是L2级量产自动驾驶的核心感知方案。

但当前2D目标检测在车载落地中仍面临多重挑战：精度与实时性矛盾突出，Transformer高精度模型算力开销大；恶劣天气、隧道明暗切换等场景图像失真，检测精度断崖式下跌；长尾未知障碍物识别能力弱；跨场景域偏移导致标注成本高昂。

未来结合视觉预训练大模型、端到端、轻量化注意力机制等，可进一步解决恶劣天气、长尾目标、跨场景适配难题，推动低成本自动驾驶规模化商用。

下期我们将继续盘点**3D目标检测**，敬请期待～**后台私信0614，免费获取目标检测算法汇总包**编辑｜咖啡鱼审核｜阿蓝前时间账号迁移，很多老粉表示咋刷不到我们文章了，的确全靠运气。既有缘相遇此文，不妨把我们星标收藏，慢慢聊车、无人机、机器人、聊技术。**欢迎关注【深蓝AI】**持续分享人工智能领域前沿动态👇
![图片](http://mmbiz.qpic.cn/sz_mmbiz_gif/943LxrS8cpCFreRWsn2fgjfEz7fB26oBpbfOsHK7zRA7xsBRS9mpSIvgQwOETOeicmb4PgKiby0nOGDo9ObI0JrvBflh4oibEdgwTEykKOSQ1w/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)
