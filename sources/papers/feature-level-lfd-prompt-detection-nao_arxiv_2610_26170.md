# 特征级示范学习（arXiv:2610.26170）

> 来源归档（paper）

- **标题：** Data-Efficient Learning from Demonstration for Humanoid Robots via Feature-Level Perception and Prompt-Based Object Detection
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.26170>
- **PDF：** <https://arxiv.org/pdf/2610.26170>
- **作者：** Keyhan Rayati, AliReza Beigy, Anoosheh Saadati, Pourya Shahverdi, Mehdi Tale Masouleh, Ahmad Kalhor, Wing-Yue Geoffrey Louie
- **arXiv 评论栏：** 6 pages, 4 figures, 2 tables. Presented at IEEE ISAIA 2026

- **入库日期：** 2026-10-26
- **一句话说明：** 基于特征级感知与提示式目标检测的人形机器人数据高效示范学习。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 从原始图像端到端学习人形操作通常需要大量示范和计算资源。
3. **周报所述创新点：** 从视频中提取人体关节、提示词检测结果和三维物体位置等紧凑特征，再预测 NAO 的关节轨迹；在极少示范条件下完成推物和抓放任务。
4. **arXiv 摘要（英文节选）：** Learning from Demonstration (LfD) allows robots to learn new skills from human examples; however, many vision-based methods require large datasets and resource-intensive end-to-end training. This paper introduces a data-efficient LfD framework for humanoid robots that learns manipulation skills from compact, task-specific features instead of raw images. The proposed approach integrates layered visual preprocessing, Kinect-based human joint estimation, prompt-based object detection, and depth information to extract human motion features and three-dimensional object locations from demonstration videos.

**对 wiki 的映射**

- [paper-feature-level-lfd-prompt-detection-nao](../../wiki/entities/paper-feature-level-lfd-prompt-detection-nao.md)
