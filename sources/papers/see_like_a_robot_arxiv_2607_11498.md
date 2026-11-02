# See like a Robot: Robot-Centric Pointmaps for VLA Models

> 来源归档（paper）

- **论文：** <https://arxiv.org/abs/2607.11498v2>（2026-09-21 修订版；已通读正文、方法、实验与结论）
- **作者：** Byungkun Lee, Dongyoon Hwang, Dongjin Kim, Hojoon Lee, Hyunseung Kim, Jaegul Choo, Minho Park
- **机构：** KAIST AI、Holiday Robotics、KRAFTON AI
- **项目页：** <https://davian-robotics.github.io/pointmap/>
- **入库日期：** 2026-11-01
- **一句话说明：** 将深度反投影到末端为原点、基座方向为坐标轴的点图，逐位置融合至 VLA 视觉特征。

## 核心摘录

- 深度与内参生成相机坐标点图，经外参变换到基座坐标并减去末端位置；保持与 RGB 像素对齐。
- 独立点图编码器由 RGB 编码器权重初始化，两路特征逐元素相加；两编码器均参与训练。
- RoboCasa 24 个任务平均成功率由 55.1% 升至 61.5%；四项真机任务平均由 37.5% 升至 70.0%，每任务 20 次试验。
- 方法依赖度量深度与相机标定；真机测试使用训练过的相机视角，不能据此断言未见视角泛化。

## 对 wiki 的映射

- [SeeR-VLA](../../wiki/entities/paper-seer-vla.md)
