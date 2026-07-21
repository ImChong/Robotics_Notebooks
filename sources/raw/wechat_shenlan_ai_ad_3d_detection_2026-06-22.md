---
title: 3D目标检测经典算法全盘点：单目、双目、激光雷达
author: 深蓝AI
date: "2026-06-22 17:32:00"
source: "https://mp.weixin.qq.com/s/1d7P4HDXmmZUZiVNx1HfXw"
---

# 3D目标检测经典算法全盘点：单目、双目、激光雷达

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAKPXr0kicZddyXdPOg1Jm7tKusPLcWicG0ALpMqpjSHZxxsu45C13rzA4XZ2leKiaxG64fPqc9zIRIj8CR43YYibVy9ic8aRib3LUd8/640?wx_fmt=png&from=appmsg#imgIndex=0)
> 「深蓝学院」推出《自动驾驶感知算法盘点》系列专栏。我们将逐一拆解感知领域的关键技术方向，为大家建立完整清晰的技术图谱。
>
> 本文是该系列的第2篇，聚焦3D检测。

---

在自动驾驶感知体系中，3D目标检测是衔接感知与决策规划的关键环节。

根据传感器配置与算法路线的不同，3D目标检测可分为三大主流方案：

- **单目方案**仅靠单个RGB摄像头完成三维感知，硬件成本最低，是经济型智驾车型的首选；
- **双目方案**通过双摄像头视差测距，以略高的硬件代价换取更优的深度精度；
- **激光雷达方案**则通过主动发射激光获取稠密点云，实现最高精度的三维感知，但硬件成本也相应更高。

本文依次梳理单目、双目、激光雷达与多模态融合三大方向的经典算法，每类介绍代表性工作的核心思想、技术特点与适用场景。

**关注并私信0622**免费获取目标检测算法汇总包👇

***深蓝AI***

**1****—******单目3D目标检测经典算法****

单目3D检测仅使用单个RGB摄像头完成图像采集与三维目标解析，硬件结构简单、部署成本低，是经济型智驾车型的主流方案。

核心难点在于：单张图像本身不包含深度信息，算法需依赖几何推理、先验数据或端到端学习来“猜测”目标距离。


**（一）FCOS3D算法**

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpC4zUSG70azT3Y4VxJnvYs4cicklb5lrRPGWIAXfRk0ib9iaibjo5pb3WO2jAdOe4EAssYZ5AaDdFaqYZq5tthUnr1xvXcCHoy8IJQ/640?wx_fmt=png&from=appmsg#imgIndex=1)

- 简介：FCOS3D是基于Anchor-Free架构的单目3D检测模型，在成熟2D检测网络基础上新增深度测距与三维尺寸估算模块，完成从二维识别到三维感知的升级。算法将语义特征与深度特征解耦处理，搭载深度不确定性损失模块，专门优化远距离目标检测的数值偏差。
- 亮点：Anchor-Free设计，无需预设候选框，缓解正负样本不均衡问题，加速模型收敛；五组并行预测分支（分类、2D框定位、深度估算、尺寸测算、角度判定）同步输出多类目标参数；模型轻量化，部署时无需新增额外车载硬件，适配算力有限的车载边缘设备。

- **优缺点：综合性能均衡，是量产车型的主流选择；结构简单，落地成本低。但目标距离超过15米时深度易漂移，强光逆光、夜间低光照工况下检测效果明显下降。**


**（二）SMOKE算法**

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpD9rB18Eu3HBwqx2m3aewkkSFGH36yVsO9cicqL7rpPQ6KkAPNYP8Es02QpNxYevdf9GqWiap4GTiboU5UefHzZRlm5bd0ZULfeaE/640?wx_fmt=png&from=appmsg#imgIndex=2)

- 简介：SMOKE是一款以精简网络结构、降低算力消耗为核心设计目标的轻量化单目3D检测算法，主动删减冗余特征分支，聚焦物体几何中心点完成三维参数推算。
- **亮点：算法借鉴人类识别远处物体时先锁定中心再判断大小与方向的经验，放弃了对目标的完整框选计算，仅需锁定车辆、行人在图像中的几何中心像素点位，再反推目标距离、物理尺寸与偏转朝向。训练阶段建立物体尺寸与深度先验数据库，推理时快速匹配物理参数，同时网络在运行过程中自动裁剪冗余特征层与通道，完成轻量化改造，使整体计算逻辑极为简洁，十分适配算力配置较低的车载处理芯片。**
- **优缺点：推理速度快、算力消耗低，常被搭载于商用车后视感知模块、车载环视泊车3D检测模块以及各类低成本无人配送小车的感知系统中，是低速简易场景的优选方案。但对远距离目标、遮挡目标的检测能力有限，精度不及FCOS3D等综合型算法，因此难以胜任高速公路或城市复杂路况下的主感知任务。**


**（三）MonoGRNet算法**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpDuiaQhic6AI1oHyzntZeF1zEVssMSjnvouiccFTtrU2soiblM3akQjLZxDbdibCB3Toyw7SHgic4icZlWub81bfs46JzrPQpia1ZE2QQY/640?wx_fmt=png&from=appmsg#imgIndex=3)

- 简介：MonoGRNet是单目3D检测中侧重几何推理的经典算法，充分结合视觉成像几何原理，依托单张二维图像还原目标三维空间信息。
- **亮点：算法不依赖海量标注数据驱动，而是挖掘图像中物体的透视关系、尺寸比例、地面接触点等隐含几何信息，通过多层几何推理网络逐步还原三维结构。其运行分为图像特征提取、几何关键点定位、三维参数推理、坐标转换四个阶段，尤其注重定位目标与地面接触的关键点（如车轮接地点），为深度估算提供可靠的几何参考，这使得算法对规则外形目标的尺寸测算精度表现较好，在单目方案中独树一帜。**
- **优缺点：对场景适应性有一定提升，且不过度依赖大规模标注数据，更多应用于算法研究与中高端车型的辅助感知模块。但在长距离、强遮挡、极端光照场景下，检测稳定性仍然存在不足。**

**后台私信0622，免费获取目标检测算法汇总包**

***深蓝AI***

**2****—******双目立体3D目标检测经典算法****

双目立体3D检测复刻了人类双眼测距的原理：左右眼接收不同视角画面，依靠画面偏差判断物体远近。双摄像头系统在同一时刻拍摄路面画面，同一障碍物在两幅图像中的像素位置会出现错位偏差，即“视差”。视差越大，目标距离越近；视差越小，目标越远。

算法依托视差数值，结合相机焦距、双摄像头安装间距等固定参数，通过物理公式求解目标深度，再融合图像语义特征完成完整3D检测。

双目方案深度测算精度远优于单目，弥补了纯视觉最大的短板。硬件成本与安装难度略高于单目，但远低于激光雷达，整体性价比突出。


**（一）3DOP算法**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpAbgJ60IicVD5qfPRc8rgrxibYRibES5k6uSonXXo7WeSYzW4K64gvzrH5NOfrEmAicMFSR2nytib2aibYa34EBjZXZrqFzs2tiar96yo/640?wx_fmt=png&from=appmsg#imgIndex=4)

- 简介：3DOP是早期双目3D目标检测领域的奠基性算法，全称为3D Object Proposals，算法诞生于双目视觉感知技术发展初期，核心作用是依托双目图像生成高质量的三维目标候选区域，为后续目标分类与参数回归提供基础支撑。
- 亮点：在双目成像与视差计算的基础上，3DOP重点优化了三维候选框的生成逻辑。传统方案大多先完成二维候选框筛选，再向三维空间拓展，容易造成空间信息丢失，而3DOP直接结合双目视差得到的深度数据，在三维空间内生成目标预选区域，让候选框从诞生之初就具备空间属性。

  算法会根据双目图像提取场景内的独立目标区域，结合深度信息划定每个目标的三维范围，再对候选区域做筛选、优化，剔除路面、天空等无效区域，保留车辆、行人、非机动车等有效目标区域。


**（二）Disp R-CNN算法**

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpDK0XEuZ9WHpNQia8cMAdkoadEyeWTYTL1WgiauXK3xXYvmay1PtT48Svicic3Nx1hUVWdw6W3e3eq1QE0QZpr7oa8xRJahbBaHoek/640?wx_fmt=png&from=appmsg#imgIndex=5)

- 简介：Disp R-CNN是目前双目立体3D检测中落地性较强的方案，针对双目视差计算的缺陷做了针对性优化。
- 亮点：传统双目算法对整幅图像做全局视差计算，当画面中出现墙面、纯色车体等纹理缺失区域时，像素匹配失效，导致深度计算错误。Disp R-CNN主动舍弃全局统一计算模式，搭载**实例级视差优化模块**：先识别图像中的目标感兴趣区域，仅针对目标所在局部区域完成像素匹配与视差计算，大幅缓解纹理缺失区域的匹配失效问题，有效提升行人、非机动车等小尺寸目标的检测效果，同时减少全域计算带来的算力浪费。
- **落地场景**：广泛搭载于配备双目摄像头的量产智驾车型、港口无人集卡、园区低速自动驾驶车辆。港口、园区等场景车速低，行人和小型作业车辆密集，双目测距优势可充分发挥，Disp R-CNN对小目标的优化特性恰好匹配这类场景需求。

**后台私信0622，免费获取目标检测算法汇总包**

***深蓝AI***

**3****—******基于点云与多模态融合的3D检测算法****

以上单目与双目方案均属“纯视觉”范畴——仅依赖RGB摄像头，不涉及激光雷达点云。当硬件条件允许搭载激光雷达时，3D检测的精度和鲁棒性会有质的飞跃。激光雷达直接提供精确的深度信息，从根本上规避了纯视觉方案“从二维图像猜测深度”的先天短板。

激光雷达点云是不规则、无序的三维坐标点集合，无法直接输入常规卷积网络。因此，点云3D检测的核心挑战在于：**如何将无序点云转化为可计算的结构化表示，同时保留足够的三维几何信息**。

围绕这一核心问题，业界发展出了**体素化、点云直接处理、点-体素混合和多模态融合四条主要技术路线。**


**（一）VoxelNet算法**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpBmHDC7ThH4RG4eMNgDIZKavoCQ96G3oSgiam3yOr4WN0msAicS2RsIxJyBH8nQfict4iaOuTPXd6GdwIic62vuRF0kw21QA8lmFiaAI/640?wx_fmt=png&from=appmsg#imgIndex=6)

- 简介：VoxelNet是点云3D检测的奠基性工作，首次实现了从原始点云到3D检测框的端到端学习。在此之前，点云处理严重依赖手工设计的特征表示（如鸟瞰图投影），而VoxelNet去除了这一需求。
- 亮点：将三维空间均匀划分为体素网格（Voxel），每个体素内通过**VFE（Voxel Feature Encoding）层**将一组点云转化为统一的特征表示。VFE层实际上是一个微型PointNet，负责提取每个体素内部的局部特征。最终整个点云被编码为一个规整的 volumetric 表示，再接入RPN完成3D检测。
- **技术特点**：VoxelNet统一了特征提取与边界框预测，是一个单阶段、端到端可训练的网络。在KITTI车辆检测基准上大幅超越了当时最先进的LiDAR方案。但缺点是3D卷积计算量巨大——体素数量庞大，大部分为空体素，算力浪费严重。


**（二）SECOND算法**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpCazsKY7kwAEowfuAplGH3ibDw9OIMs92VPCWeOcRKTXoONlHdnvGgrrPy5hSRibPx4Hia0B9OxcmMJ0aVRu1ic6uQw4usQZ3vS58w/640?wx_fmt=png&from=appmsg#imgIndex=7)

- 简介：SECOND（Sparsely Embedded Convolutional Detection）是VoxelNet的直接优化版本，核心创新是引入**3D稀疏卷积**。其洞察很简单：LiDAR点云中绝大多数体素是空的，传统3D卷积会遍历全部网格造成算力浪费，而稀疏卷积仅对非空体素进行计算。
- **亮点**：在VoxelNet框架基础上，将稠密3D卷积替换为稀疏卷积和子流形稀疏卷积。只计算包含点云的非空体素，跳过空白区域，大幅降低计算量与内存占用。
- **技术特点**：在几乎不损失精度的前提下大幅提升推理速度，改善了VoxelNet实时性差的主要痛点。SECOND结构高效、稳定性强，成为后续多数体素类3D检测算法的基础框架。


**（三）PointPillars算法**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpCR5rNzFFWUN0CpLRDGicQpHpY8iaqNGODacPg03OVWMTWv1oia0Mu3Q4lxUpcnQ3fUxIicEjL1kk9fTia1Af36FDA2nCOT3XLgZaI8/640?wx_fmt=png&from=appmsg#imgIndex=8)

- 简介：PointPillars是目前**工业界量产应用最广的实时3D检测算法**，核心创新是放弃立方体体素，改用竖直柱状结构（Pillars）划分点云空间。
- **亮点**：在鸟瞰平面上划分网格，每个网格向上拉伸为竖直柱子，收纳该区域全部高度的点云。通过轻量PointNet提取每根柱子的特征，将三维空间信息压缩为二维伪图像（Pseudo-image），再通过常规2D卷积完成检测。
- **技术特点**：设计规避了3D卷积的海量计算，推理速度极快——仅用LiDAR即可在KITTI基准上超越融合方法，运行速度达62Hz，轻量版可达105Hz。精度与速度均衡，是量产车型的首选方案。


**（四）PointRCNN算法**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpAxep6YSkMb2vVN9HyKsddyxDqEjV2NVFAwc9uOcwVMlTDFLAlFicBq83KXts4ZAxBbrnlBRSw4Fq7LjnicoKrJy7RGcmm7CcYrQ/640?wx_fmt=png&from=appmsg#imgIndex=9)

- 简介：PointRCNN是高精度两阶段点云3D检测算法，核心优势是**最大程度保留原始点云的精准几何信息**。
- **亮点**：第一阶段直接对原始点云进行前景/背景分割，从点云中自底向上生成少量高质量3D候选框——不做量化压缩，保留完整的物体轮廓、位置与尺寸细节。第二阶段将每个候选框内的点云变换到规范坐标系，融合第一阶段的全局语义特征，完成边框精修与置信度预测。
- **技术特点**：检测精度高，但对算力消耗大，推理速度较慢。


**（五）AVOD算法**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpDEUiaShh38uAqiawbUgxanFVeJeHT4sxmbne8JrhZbe5zOWmRVQcMiaGbSicaRbnkLH74AsKQ98yDj6Gyxu1M1J6SKCyzhqJ746oc/640?wx_fmt=png&from=appmsg#imgIndex=10)

- 简介：AVOD（Aggregate View Object Detection）是早期经典的多视图融合3D检测算法。
- **亮点**：同时利用LiDAR点云和RGB图像两种模态。将点云投影为鸟瞰图（BEV）和前视图（FV），分别提取特征后与图像特征融合。两个子网络共享特征：RPN生成3D候选框，二级检测网络完成精准的3D边框回归与分类。
- **技术特点**：早期多视图融合的代表作，兼顾空间几何信息与纹理细节。在KITTI 3D检测基准上达到当时的SOTA，且运行速度满足实时要求、内存占用低。


**（六）Frustum PointNets算法**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAXkXGDIVXbT3HEj9DgBgVJibia7libgfhBvmREAXrLZgB27sKupv3RexgHPFxaX5ObIPKvPqkT7GcqOQibKic0D0ddGbX99SsJBHUU/640?wx_fmt=png&from=appmsg#imgIndex=11)

- 简介：Frustum PointNets由PointNet作者Charles R. Qi等人提出，开创了**“2D驱动3D”**的多模态融合范式。
- **亮点**：先用成熟的2D检测器在RGB图像中定位目标，生成2D检测框；再根据相机投影参数，将该2D框“拉升”为三维空间中的一个**视锥（Frustum）**；最后在视锥对应的点云子集内，用PointNet完成3D检测框的回归。
- **技术特点**：巧妙地将3D空间中的搜索范围缩小到2D检测框对应的视锥区域，大幅降低计算复杂度。直接操作原始点云，在强遮挡或点云稀疏场景下仍能精确估计3D边界框。在KITTI和SUN RGB-D基准上均大幅超越当时SOTA。


**（七）LiDAR-RCNN算法**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpCf2EBfuj2FTFO2CJar6cz5EEfelCnLzOcUGAeiaEOSOeLDkdAyPQ0oGN0ZRyrTXOErElkM2QzT3ZRPR2xSyR7ZfgicRLjgVptlI/640?wx_fmt=png&from=appmsg#imgIndex=12)

- 简介：LiDAR-RCNN是一个**通用的两阶段精修检测器**，可以插入任何现有的3D检测器之后进一步提升精度。
- **亮点**：采用基于点的方法而非流行的体素方法。核心洞察是：直接用PointNet等点云处理方法时，模型会**忽略候选框的尺寸信息**——网络只看到点云本身，不知道这些点来自多大的框。LiDAR-RCNN针对此问题提出多种修正方法。
- **技术特点**：可插入任意现有3D检测器提升性能。在Waymo Open Dataset和KITTI上均验证了通用性与优越性。基于PointPillars变体即可达到新的SOTA，且额外开销很小。


**（八）CenterPoint算法**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpC2m9wh9v4NIOgTgGUMmzdsdMtubwlEsQSibu0GTFFUXkWKictkmNkYPWfibcIQVLkiaeQaKHNKkCVXHbkdhC6h1JgPTd6VHJGElFg/640?wx_fmt=png&from=appmsg#imgIndex=13)

- 简介：CenterPoint是当前3D检测领域**最具影响力的算法之一**，将2D检测中的CenterNet思想成功迁移到3D点云。
- **亮点**：将3D目标检测与跟踪统一为**“以点为中心”**的范式。第一阶段用关键点检测器检测物体的中心点，并回归3D尺寸、3D朝向和速度等属性；第二阶段利用物体上的额外点特征对这些估计进行精修。3D目标跟踪简化为贪心的最近点匹配。
- **技术特点**：无需NMS、结构简洁、速度快且精度高。在nuScenes和Waymo数据集上大幅超越此前方法。在Waymo Open Dataset上相比此前SOTA提升10-20%，运行速度达13FPS。

**后台私信0622，免费获取目标检测算法汇总包**

***深蓝AI***

**4****—******写在最后****

单目、双目、激光雷达三条路线各有用武之地，不存在谁取代谁。

工程师在做方案选型时，看的是车型定位、传感器配置和场景需求之间的匹配度，而不是哪条路线更“先进”。

本文逐一拆解了各算法的核心思想、技术特点和落地定位，希望能为从事感知算法研发的工程师们提供一个快速对照参考。

编辑｜咖啡鱼审核｜阿蓝前时间账号迁移，很多老粉表示咋刷不到我们文章了，的确全靠运气。既有缘相遇此文，不妨把我们星标收藏，慢慢聊车、无人机、机器人、聊技术。**欢迎关注【深蓝AI】**持续分享人工智能领域前沿动态👇![图片](http://mmbiz.qpic.cn/sz_mmbiz_gif/943LxrS8cpCFreRWsn2fgjfEz7fB26oBpbfOsHK7zRA7xsBRS9mpSIvgQwOETOeicmb4PgKiby0nOGDo9ObI0JrvBflh4oibEdgwTEykKOSQ1w/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)
