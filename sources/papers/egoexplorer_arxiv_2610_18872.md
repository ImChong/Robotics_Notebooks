# EgoExplorer：可穿戴视觉–动作–触觉人类移动操作数据集

> 来源归档（ingest）— 根据 arXiv v1 全文、作者项目页、数据集卡和工具仓库说明整理；非原文镜像。

- **论文标题：** EgoExplorer: A Vision-Motion-Touch Observatory for Human Loco-Manipulation In Situ
- **类型：** paper / dataset / egocentric data / human loco-manipulation
- **作者：** Yunsong Zhou、Xuekun Jiang、Haiqi Shen、Hangxu Liu、Chaojian Shi、Yujie Liu、Jingyuan Li、Haoran Yang、Jiafei Cao、Dongzhan Zhou、Dong Wang、Yao Mu、Chunhua Shen、Weinan Zhang
- **机构：** 上海人工智能实验室；浙江大学；上海交通大学（论文作者标注）
- **日期：** arXiv v1 提交于 2026-10-13（cs.RO、cs.CV）
- **论文：** [arXiv 摘要](https://arxiv.org/abs/2610.18872) · [HTML 全文](https://arxiv.org/html/2610.18872v1) · [PDF](https://arxiv.org/pdf/2610.18872)
- **项目页：** <https://internrobotics.github.io/egoexplorer.github.io/>
- **数据集：** <https://huggingface.co/datasets/InternRobotics/EgoExplorer>（需申请访问）
- **采集与重建工具仓：** <https://github.com/InternRobotics/EgoExplorer>（仓库说明工具代码预计 2026 年 11 月初发布）
- **一句话说明：** 用头戴相机、五个 inside-out 跟踪器与手套在未布置基站的真实空间采集连续人类移动操作；经 C³（Calibrate–Complete–Constrain）从稀疏身体测量重建全身动作，并对齐第一视角视频、手指动作和（WUJI 手套配置下的）区域触觉。

## 数据与传感配置

论文报告 **2,715 段、100 小时**录制，覆盖 10 类任务模板与 5 个环境，主要包括实验室专业流程，也含家庭/工作空间任务。每段对齐第一视角视频、重建的全身 SMPL-X 动作、手指关节动作与步骤级描述。

可穿戴采集系统包括 GoPro HERO13 头戴相机和 5 个 VIVE Ultimate Tracker（头部、双腕、双踝）。跟踪器使用 inside-out 定位，因此不需在房间安装外部基站或保持外部光学视线。两种手套配置可互换：MANUS Metagloves 记录手指动作；WUJI Glove 记录手指动作并额外提供区域触觉。**触觉不是每条录制都有：**论文附录报告常规录制中 WUJI 流约占 41%，MANUS 约占 58%。

## C³：从稀疏追踪到全身运动

1. **Calibrate：** 通过规定动作校准个体体型、尺度及传感器安装变换，先把固定安装偏差与关节运动分离。
2. **Complete：** 使用时序动作先验，从五个 tracker 锚点补全未直接观测的关节运动。
3. **Constrain：** 联合测量一致性、人体表面间隙与地面约束优化整段轨迹，减少仅匹配 tracker 点却出现身体穿透或脚底漂浮的情况。

在论文评测子集上，身体点误差为 **18.0 mm**，相对该研究中最强基线 AvatarPoser（五锚点重训，64.1 mm）降低 **71.9%**。这是一项特定参考标注和评测子集下的结果，不代表所有录制片段都有同等真值精度。

## 开放状态与使用边界（核对日期：2026-10-15）

- Hugging Face 数据集卡显示数据需接受访问条件并提交申请；当时提供 **18 段、约 66 分钟**的一小时 mini release，覆盖 10 个任务模板且包含四类数据流。完整 100 小时集需加入 waitlist，不能把论文总规模理解成已可匿名下载。
- 数据集卡标注 **CC BY-NC 4.0**；使用前应阅读最新访问条款。
- 作者工具仓库当前主要为说明与素材；README 称采集/重建工具代码预计 **2026 年 11 月初**发布。应区分公开项目页面、受限数据访问与尚待发布的工具实现。
- 论文还做了多模态生成和人形全身动作重定向的初步仿真研究；这是初步可行性探索，不等于已验证的真机策略或完整部署栈。

## 关联页面

- [EgoExplorer 详情页](../../wiki/entities/paper-egoexplorer-vision-motion-touch-dataset.md)
- [人形机器人数据采集产业地图](../../wiki/queries/humanoid-robot-data-collection-landscape.md)
- [Manipulation 任务页](../../wiki/tasks/manipulation.md)

## 参考链接

- [论文 PDF](https://arxiv.org/pdf/2610.18872)
- [作者项目页](https://internrobotics.github.io/egoexplorer.github.io/)
- [Hugging Face 数据集卡](https://huggingface.co/datasets/InternRobotics/EgoExplorer)
- [采集与重建工具仓](https://github.com/InternRobotics/EgoExplorer)
