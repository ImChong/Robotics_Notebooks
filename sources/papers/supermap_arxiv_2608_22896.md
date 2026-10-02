# SuperMap（RSS 2026）

> 来源归档（ingest），基于官方项目页、RSS 摘要和官方仓库核查。

- **标题：** SuperMap: A Spatio-Temporal SLAM System for Visual-Language Navigation
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）AirLab
- **作者：** Shibo Zhao、Guofei Chen、Honghao Zhu、Zhiheng Li、Changwei Yao、Nader Zantout、Seungchan Kim、Wenshan Wang、Ji Zhang、Sebastian Scherer
- **会议：** RSS 2026
- **arXiv：** <https://arxiv.org/abs/2608.22896>
- **论文：** <https://www.roboticsproceedings.org/rss22/p052.pdf>
- **项目页：** <https://superodometry.com/supermap>
- **代码入口：** <https://github.com/superxslam/SuperMap>
- **核查日期：** 2026-10-02
- **开放状态：** 待发布可运行源码；当前公开仓库为 README 与 doc 文档，README 仍注明代码将在 RSS 后发布。项目页宣称 open-source，不能据此认定实现已发布。

## 核心资料摘录

1. 高频几何 SLAM 与异步开放词汇感知分工：几何提供位姿与三维锚点，感知提供物体类别与分割。无需针对场景重新训练，但依赖预训练视觉模型。
2. 通过三维实例关联、重新激活和存在/标签置信度更新，维持遮挡与场景变化中的物体身份，清理过时语义。
3. 查询式 4D 场景图记录物体语义、空间关系和历史，使 VLM 能查询移动过的物体及过去场景；这是显式空间记忆，不是预测视频的生成式世界模型。

## 评测摘录

项目页 ScanNet 表：SuperMap mIoU 27.42%、f-mIoU 43.50%、Acc 55.48%；HOV-SG 分别为 26.79%、36.05%、35.17%。时空变化检测表中出现事件 recall：Bucket 1.000、Cart 0.262、Sign 0.583，因此不能概括成所有出现事件均完美检测。校园两小时演示证明持续运行展示，不等同于标准 ObjectNav 成功率。

## 对 wiki 的映射

- [SuperMap 论文实体](../../wiki/entities/paper-supermap.md)
- [视觉语言导航](../../wiki/tasks/vision-language-navigation.md)：加入外部空间记忆路线。

## 项目与仓库归档

- [项目页](../sites/supermap.md)
- [官方仓库](../repos/supermap.md)
