# 图宾根大学《Virtual Humans》课程

> 来源归档

- **标题：** Virtual Humans
- **类型：** course
- **课程主页：** <https://virtualhumans.mpi-inf.mpg.de/VH26/>
- **公开讲义示例：** [Lecture 01 · Organization and Introduction to Body Models（PDF）](https://virtualhumans.mpi-inf.mpg.de/VH26/slides/Lecture_01_0_Organization.pdf)
- **主讲：** Prof. Dr. Gerard Pons-Moll
- **机构：** 图宾根大学（University of Tübingen）；Real Virtual Humans 研究组
- **学期：** 2026/27 冬季学期
- **课程编号 / 学分：** ML4508 / 9 ECTS
- **源码仓库：** 官网未列课程专属 GitHub 或代码仓库；截至 2026-10-15，图片中提到的“开源工具箱”尚未能从课程官网核实。
- **入库日期：** 2026-10-15
- **一句话说明：** 一门从人体参数化模型与几何处理讲到神经渲染、行为合成和人类技能迁移到机器人的研究生课程。

## 课程内容

课程围绕从 3D 扫描、图像和视频学习虚拟人体，覆盖经典网格与纹理表示，以及用神经网络表示外观和行为。官网列出的主题包括：

- 身体与服装模型、旋转表示和曲面参数化。
- ICP、非刚性对齐，以及将 SMPL 拟合到扫描、图像和 IMU 数据。
- 神经隐式表示、点表示、NeRF 与 Gaussian Splatting 人体模型。
- 人类行为捕捉与合成，包括动作扩散模型。
- 基于物理的方法、策略学习，以及人类技能向机器人的迁移。

课程安排为每周一次讲座和一次辅导，另有三次双周作业与研究项目。先修要求包括线性代数、概率、优化和 Python 编程；PyTorch 与计算机图形学经验有帮助。

## 为什么值得保留

这门课把数字人体建模链路和机器人学习联系起来：人体模型与动作捕捉提供可动画的人体表示，行为合成扩展动作先验，最后讨论物理可行性、策略学习和迁移到机器人。对机器人开发者而言，它适合作为理解“人体数据如何变成机器人可用动作”的跨学科学习入口；课程本身不是机器人控制器或可直接部署的机器人软件。

## 开源与资料状态

课程官网目前提供课程安排及讲义入口，首讲的组织说明 PDF 可直接访问；页面未列出课程专属代码、数据集或工具箱仓库。因此，这里只归档课程与公开讲义，不把图片中的“开源工具箱”表述登记为已验证的源码项目。若官网后续发布代码地址，可再补充对应来源记录。

## 对 wiki 的映射

- [GMR：通用动作重定向](../../wiki/methods/motion-retargeting-gmr.md) — 人体动作转为机器人关节轨迹的几何层。
- [基于扩散模型的运动生成](../../wiki/methods/diffusion-motion-generation.md) — 课程涉及的人体行为合成与动作扩散模型。

## 推荐继续阅读（外部）

- [Virtual Humans · 2026/27 冬季学期课程主页](https://virtualhumans.mpi-inf.mpg.de/VH26/)
- [Lecture 01 · Organization and Introduction to Body Models（PDF）](https://virtualhumans.mpi-inf.mpg.de/VH26/slides/Lecture_01_0_Organization.pdf)
