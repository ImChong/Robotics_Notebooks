---
type: overview
tags:
  - dataset
  - data-panorama
status: draft
updated: 2026-10-20
summary: "2026-10-19 公众号综述的 85 个具身智能数据集全景：逐项映射到唯一 wiki 节点，41 项复用已有节点、44 项新建节点。"
related:
  - ../concepts/open-x-embodiment.md
  - ../concepts/embodied-scaling-laws.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# 具身智能数据集全景（85 项去重映射）

## 一句话定义

按数据形态与用途，把 85 个公开具身数据集及集合分为真实机器人、仿真与合成、人类示范与第一视角、导航与三维场景、灵巧手与抓取、触觉与多模态六组，并**每个数据集对应唯一 wiki 节点**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作模型，多数真机与仿真数据的主要消费方 |
| Sim2Real | Simulation to Real | 仿真到真实迁移，合成数据类节点的核心问题 |
| UMI | Universal Manipulation Interface | 手持夹爪式人类示范采集接口，真机数据的重要来源 |
| RGB-D | RGB + Depth | 彩色加深度图像，导航与抓取数据的常见模态 |
| LeRobot | LeRobot 机器人学习库 | 社区数据格式与工具链，多个节点采用其结构 |

## 为什么重要

- 数据集名称与规模差异大，难以判断“同一数据集”是否重复出现；本页用**一对一映射**避免同一资源在 wiki 中多次建页。
- 同一项目的论文、数据与代码仍归入同一实体节点（沿用 [ingest 规范](../../schema/ingest-workflow.md) 的项目去重规则）；已有节点的数据集以复用为准。
- 文章收录口径、分组与数字均为二手转述，见来源归档：[wechat-embodied-dataset-panorama-2026-10-19](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)。

## 核心原理

本页不新增方法结论，只做三件事：分组、去重映射、给出每个数据集的入口。分组依据数据形态（真机、仿真、人类视频、三维场景、手物交互、触觉），而非机构或年份。

## 工程实践

| 项 | 说明 |
|----|------|
| 选数据 | 先按任务类型（操作 / 导航 / 手物 / 触觉）在对应分组中筛选，再看节点的许可与规模 |
| 去重 | 新增同一数据集时先检索本页与 `wiki/entities/` 的 arXiv 编号、项目地址，命中则复用 |
| 更新 | 文章或官方页数字变化时，以官方页为准修改节点，并更新本页映射 |

## 局限与风险

- 收录口径为综述作者选定，并非完整清单；“集合成员不重复计数”意味着部分集合（如 Ego-Exo4D 与 Ego4D 系列）的子数据未单列。
- 复用的 41 项节点写于不同时间，其内容未因本次综述同步修改；新建的 44 项节点事实来自综述，**未逐项核对原论文**。
- 许可与获取方式会随版本变化（部分数据需申请或签协议），使用前以官方页为准。

## 关联页面

- [开放 X 具身数据（OXE）](../concepts/open-x-embodiment.md)：跨机构真机数据的合并范式
- [具身智能的数据规模律](../concepts/embodied-scaling-laws.md)：数据规模与泛化的讨论
- [VLA 方法](../methods/vla.md)：多数真机与合成数据的训练对象
- [操作任务](../tasks/manipulation.md)：多数数据集所服务的任务

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [Open X-Embodiment 项目页](https://robotics-transformer-x.github.io/)
- [LeRobot 数据集格式文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## 85 项映射

### 真实机器人（22 项）

| # | 数据集 | 节点 | 去重状态 |
|---|--------|------|----------|
| 01 | Open X-Embodiment（OXE） | [paper-open-x-embodiment](../entities/paper-open-x-embodiment.md) | 复用已有节点 |
| 02 | DROID | [paper-rcl-ref-aa9c28fbf4242ea03696-droid-a-large-scale-in-the-wild-robot-manipulati](../entities/paper-rcl-ref-aa9c28fbf4242ea03696-droid-a-large-scale-in-the-wild-robot-manipulati.md) | 复用已有节点 |
| 03 | BridgeData V2 | [painode-085-bridgedatav2](../entities/painode-085-bridgedatav2.md) | 复用已有节点 |
| 04 | LeRobot Community Dataset v3 | [lerobot-dataset-v3](../concepts/lerobot-dataset-v3.md) | 复用已有节点 |
| 05 | AgiBot World（Colosseo / 2026） | [paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation](../entities/paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation.md) | 复用已有节点 |
| 06 | RoboMIND（含 2.0） | [painode-094-robomind](../entities/painode-094-robomind.md) | 复用已有节点 |
| 07 | RH20T | [painode-092-rh20t](../entities/painode-092-rh20t.md) | 复用已有节点 |
| 08 | ALOHA Demonstration Dataset（ACT） | [paper-act](../entities/paper-act.md) | 复用已有节点 |
| 09 | FastUMI / FastUMI-100K | [dataset-fastumi](../entities/dataset-fastumi.md) | 新建节点 |
| 10 | HiFi-UMI-2K | [paper-hifi-umi](../entities/paper-hifi-umi.md) | 复用已有节点 |
| 11 | Hy-Embodied-0.5-VLA-Data | [dataset-hy-embodied-0-5-vla-data](../entities/dataset-hy-embodied-0-5-vla-data.md) | 新建节点 |
| 12 | Galaxea Open-World Dataset | [dataset-galaxea-open-world](../entities/dataset-galaxea-open-world.md) | 新建节点 |
| 13 | Fourier ActionNet Dataset | [dataset-fourier-actionnet](../entities/dataset-fourier-actionnet.md) | 新建节点 |
| 14 | UnifoLM-WBT Dataset | [dataset-unifolm-wbt](../entities/dataset-unifolm-wbt.md) | 新建节点 |
| 15 | Humanoid Everyday | [humanoid-everyday-dataset](../entities/humanoid-everyday-dataset.md) | 复用已有节点 |
| 16 | AIST Bimanual Manipulation Dataset | [dataset-aist-bimanual-manipulation](../entities/dataset-aist-bimanual-manipulation.md) | 新建节点 |
| 17 | RoboCOIN | [dataset-robocoin](../entities/dataset-robocoin.md) | 新建节点 |
| 18 | MolmoAct2-BimanualYAM | [paper-molmoact2](../entities/paper-molmoact2.md) | 复用已有节点 |
| 19 | FineART | [paper-fineart-vla](../entities/paper-fineart-vla.md) | 复用已有节点 |
| 20 | ARIO（All Robots In One） | [dataset-ario](../entities/dataset-ario.md) | 新建节点 |
| 21 | HABIT | [dataset-habit](../entities/dataset-habit.md) | 新建节点 |
| 22 | ABC-130K | [dataset-abc-130k](../entities/dataset-abc-130k.md) | 新建节点 |

### 仿真与合成（17 项）

| # | 数据集 | 节点 | 去重状态 |
|---|--------|------|----------|
| 23 | robomimic Datasets | [robomimic](../entities/robomimic.md) | 复用已有节点 |
| 24 | LIBERO（含 LIBERO-Plus） | [libero-benchmark](../entities/libero-benchmark.md) | 复用已有节点 |
| 25 | CALVIN | [calvin-benchmark](../entities/calvin-benchmark.md) | 复用已有节点 |
| 26 | ManiSkill Demonstrations | [maniskill2](../entities/maniskill2.md) | 复用已有节点 |
| 27 | MimicGen Datasets | [mimicgen](../entities/mimicgen.md) | 复用已有节点 |
| 28 | BEHAVIOR-1K Dataset | [behavior-1k](../entities/behavior-1k.md) | 复用已有节点 |
| 29 | RoboCasa365 | [paper-notebook-robocasa365-a-large-scale-simulation-framework-f](../entities/paper-notebook-robocasa365-a-large-scale-simulation-framework-f.md) | 复用已有节点 |
| 30 | RoboTwin 2.0 | [robotwin](../entities/robotwin.md) | 复用已有节点 |
| 31 | PhysicalAI-Robotics-GR00T-X-Embodiment-Sim | [isaac-gr00t](../entities/isaac-gr00t.md) | 复用已有节点 |
| 32 | VIMA-Data（VIMA-Bench） | [paper-pai-2210-03094-vima](../entities/paper-pai-2210-03094-vima.md) | 复用已有节点 |
| 33 | VLABench | [painode-024-vlabench](../entities/painode-024-vlabench.md) | 复用已有节点 |
| 34 | BiGym Demonstrations（含 2.0） | [paper-notebook-bigym-a-demo-driven-mobile-bi-manual-manipulatio](../entities/paper-notebook-bigym-a-demo-driven-mobile-bi-manual-manipulatio.md) | 复用已有节点 |
| 35 | ARNOLD | [painode-010-arnold](../entities/painode-010-arnold.md) | 复用已有节点 |
| 36 | RoboVerse Data | [dataset-roboverse-data](../entities/dataset-roboverse-data.md) | 新建节点 |
| 37 | HumanoidMimicGen Dataset | [paper-humanoidmimicgen](../entities/paper-humanoidmimicgen.md) | 复用已有节点 |
| 38 | InternData-A1 | [dataset-internData-a1](../entities/dataset-internData-a1.md) | 新建节点 |
| 39 | MolmoBot-Data | [dataset-molmobot-data](../entities/dataset-molmobot-data.md) | 新建节点 |

### 人类示范与第一视角（7 项）

| # | 数据集 | 节点 | 去重状态 |
|---|--------|------|----------|
| 40 | Ego4D | [paper-ego4d](../entities/paper-ego4d.md) | 复用已有节点 |
| 41 | Ego-Exo4D | [dataset-ego-exo4d](../entities/dataset-ego-exo4d.md) | 新建节点 |
| 42 | EPIC-KITCHENS-100 | [painode-089-epickitchens100](../entities/painode-089-epickitchens100.md) | 复用已有节点 |
| 43 | AMASS | [amass](../entities/amass.md) | 复用已有节点 |
| 44 | HoloAssist | [dataset-holoassist](../entities/dataset-holoassist.md) | 新建节点 |
| 45 | EgoScale | [egoscale](../methods/egoscale.md) | 复用已有节点 |
| 46 | Embody 3D | [dataset-embody-3d](../entities/dataset-embody-3d.md) | 新建节点 |

### 导航与三维场景（16 项）

| # | 数据集 | 节点 | 去重状态 |
|---|--------|------|----------|
| 47 | R2R（Room-to-Room） | [paper-vln-01-r2r](../entities/paper-vln-01-r2r.md) | 复用已有节点 |
| 48 | RxR（Room-Across-Room） | [paper-rxr](../entities/paper-rxr.md) | 复用已有节点 |
| 49 | REVERIE | [paper-vln-03-reverie](../entities/paper-vln-03-reverie.md) | 复用已有节点 |
| 50 | CVDN（Cooperative Vision-and-Dialog Navigation） | [dataset-cvdn](../entities/dataset-cvdn.md) | 新建节点 |
| 51 | SOON（FAO 数据集） | [dataset-soon](../entities/dataset-soon.md) | 新建节点 |
| 52 | ALFRED | [painode-009-alfred](../entities/painode-009-alfred.md) | 复用已有节点 |
| 53 | TEACh | [painode-023-teach](../entities/painode-023-teach.md) | 复用已有节点 |
| 54 | Matterport3D | [paper-rcl-1709-06158-matterport3d-learning-from-rgb-d-data-in-indoor](../entities/paper-rcl-1709-06158-matterport3d-learning-from-rgb-d-data-in-indoor.md) | 复用已有节点 |
| 55 | HM3D（Habitat-Matterport 3D） | [dataset-hm3d](../entities/dataset-hm3d.md) | 新建节点 |
| 56 | ScanNet | [dataset-scannet](../entities/dataset-scannet.md) | 新建节点 |
| 57 | Gibson Environment Dataset | [dataset-gibson-environment](../entities/dataset-gibson-environment.md) | 新建节点 |
| 58 | HSSD / HSSD-200 | [dataset-hssd](../entities/dataset-hssd.md) | 新建节点 |
| 59 | ReplicaCAD | [dataset-replicacad](../entities/dataset-replicacad.md) | 新建节点 |
| 60 | EmbodiedScan | [dataset-embodiedscan](../entities/dataset-embodiedscan.md) | 新建节点 |
| 61 | ProcTHOR-10K | [dataset-procthor-10k](../entities/dataset-procthor-10k.md) | 新建节点 |
| 62 | PartNet-Mobility（SAPIEN） | [dataset-partnet-mobility](../entities/dataset-partnet-mobility.md) | 新建节点 |

### 灵巧手与抓取（12 项）

| # | 数据集 | 节点 | 去重状态 |
|---|--------|------|----------|
| 63 | GraspNet-1Billion | [painode-218-graspnet1billion](../entities/painode-218-graspnet1billion.md) | 复用已有节点 |
| 64 | DexGraspNet | [paper-rcl-2210-02697-dexgraspnet-a-large-scale-robotic-dexterous-gras](../entities/paper-rcl-2210-02697-dexgraspnet-a-large-scale-robotic-dexterous-gras.md) | 复用已有节点 |
| 65 | SynGrasp-1B / GraspVLA | [cn-os-graspvla](../entities/cn-os-graspvla.md) | 复用已有节点 |
| 66 | EgoDex | [paper-notebook-egodex-learning-dexterous-manipulation-from-larg](../entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md) | 复用已有节点 |
| 67 | DexYCB | [dataset-dexycb](../entities/dataset-dexycb.md) | 新建节点 |
| 68 | OakInk / OakInk2 | [dataset-oakink](../entities/dataset-oakink.md) | 新建节点 |
| 69 | HO-3D（含 H2O-3D、HO-3D v3） | [dataset-ho-3d](../entities/dataset-ho-3d.md) | 新建节点 |
| 70 | H2O Dataset（2 Hands and Objects） | [dataset-h2o](../entities/dataset-h2o.md) | 新建节点 |
| 71 | HOI4D | [dataset-hoi4d](../entities/dataset-hoi4d.md) | 新建节点 |
| 72 | HOT3D | [dataset-hot3d](../entities/dataset-hot3d.md) | 新建节点 |
| 73 | ARCTIC | [dataset-arctic](../entities/dataset-arctic.md) | 新建节点 |
| 74 | GRAB | [dataset-grab](../entities/dataset-grab.md) | 新建节点 |

### 触觉与多模态（11 项）

| # | 数据集 | 节点 | 去重状态 |
|---|--------|------|----------|
| 75 | Touch and Go | [dataset-touch-and-go](../entities/dataset-touch-and-go.md) | 新建节点 |
| 76 | ObjectFolder 2.0（仿真数据） | [dataset-objectfolder-2-0](../entities/dataset-objectfolder-2-0.md) | 新建节点 |
| 77 | ObjectFolder Real | [dataset-objectfolder-real](../entities/dataset-objectfolder-real.md) | 新建节点 |
| 78 | Touch100k | [dataset-touch100k](../entities/dataset-touch100k.md) | 新建节点 |
| 79 | OpenTouch | [dataset-opentouch](../entities/dataset-opentouch.md) | 新建节点 |
| 80 | Touch in the Wild | [dataset-touch-in-the-wild](../entities/dataset-touch-in-the-wild.md) | 新建节点 |
| 81 | TacQuad（AnyTouch 配套） | [paper-anytouch](../entities/paper-anytouch.md) | 复用已有节点 |
| 82 | VisGel | [dataset-visgel](../entities/dataset-visgel.md) | 新建节点 |
| 83 | YCB-Slide | [dataset-ycb-slide](../entities/dataset-ycb-slide.md) | 新建节点 |
| 84 | YCB-Sight | [dataset-ycb-sight](../entities/dataset-ycb-sight.md) | 新建节点 |
| 85 | RCT（Robotic Contact Tactile） | [dataset-rct-robotic-contact-tactile](../entities/dataset-rct-robotic-contact-tactile.md) | 新建节点 |

