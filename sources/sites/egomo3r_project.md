# EgoMo3R 官方项目主页

> 官方项目资料归档

- **标题：** EgoMo3R: Joint Egocentric Motion and Scene Reconstruction / 第一人称人体运动与三维场景联合重建
- **类型：** site / project
- **来源：** Chaitanya Patel 等，斯坦福大学与松下
- **项目：** https://chaitanya100100.github.io/EgoMo3R/
- **论文：** https://arxiv.org/abs/2611.01627（[原文归档](../papers/egomo3r_arxiv_2611_01627.md)）
- **代码：** https://github.com/chaitanya100100/EgoMo3R（[源码归档](../repos/egomo3r_chaitanya100100.md)）
- **运动先验代码：** https://github.com/chaitanya100100/motion_diffusion
- **互动结果：** https://chaitanya100100.github.io/EgoMo3R/interactive/
- **核对日期：** 2026-11-15
- **核对范围：** 官方主页作者/机构、NeurIPS 2026、abstract、method、资源入口、BibTeX；核对官方源码仓库 README 与独立运动先验代码。
- **通读：** 已完成
- **沉淀到 wiki：** [EgoMo3R](../../wiki/entities/paper-egomo3r.md)

## 项目页要点

- 方法图明确：输入单目 egocentric 视频，利用单目深度/光流/点轨迹进行 **Scene Refinement**；扩散先验在相机轨迹条件下进行 **Motion Alignment**；交替优化 C、S、M。
- 输出：相机轨迹、度量尺度点图与人体运动，同处重力对齐的世界坐标系。
- **开源状态：已开源（研究推理/优化实现 + 示例数据 + 单独预训练人体运动先验）**；不是即时部署 SDK，具体依赖和相机内参限制见源码归档。
- 未发现项目页声称已验证机器人控制策略训练或真机部署；机器人数据用途属于潜在应用，非本研究完成的实验。

## 官方链接

- [项目主页](https://chaitanya100100.github.io/EgoMo3R/)
- [官方论文](https://arxiv.org/html/2611.01627)
- [主仓库](https://github.com/chaitanya100100/EgoMo3R)
- [motion_diffusion 仓库](https://github.com/chaitanya100100/motion_diffusion)
