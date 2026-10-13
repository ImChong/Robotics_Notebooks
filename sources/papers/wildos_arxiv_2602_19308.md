# WildOS 论文来源归档

## 元数据

- **题目：** WildOS: Open-Vocabulary Object Search in the Wild
- **作者：** Hardik Shah, Erica Tevere, Deegan Atha, Marcel Kaufmann, Shehryar Khattak, Manthan Patel, Marco Hutter, Jonas Frey, Patrick Spieler
- **单位：** NASA 喷气推进实验室（JPL）；ETH Zurich Robotics Systems Lab；Stanford University；University of California, Berkeley
- **预印本：** https://arxiv.org/abs/2602.19308
- **项目页：** https://leggedrobotics.github.io/wildos/（归档见 [wildos-project.md](../sites/wildos-project.md)）
- **代码：** https://github.com/nasa-jpl/nebula2-wildos（归档见 [nebula2-wildos.md](../repos/nebula2-wildos.md)）
- **数据：** https://huggingface.co/datasets/leggedrobotics/wildos（归档见 [wildos-frontiers.md](../datasets/wildos-frontiers.md)）
- **发表状态：** 项目页注明已被 IEEE Transactions on Field Robotics（T-FR）接收
- **提交日期：** 2026-02-22
- **入库 / 核查日期：** 2026-10-13

## 研究问题

户外非结构化环境中，机器人通常只能依赖深度传感器在近处建图。传统几何 frontier 会把探索引向地图边缘，却难判断远处方向是否可通行、是否通向语言描述的目标。WildOS 把视觉语义判断接到几何导航图上，使机器人在保留空间记忆和局部安全约束的同时，进行长距离开放词汇目标搜索。

## 方法摘录

1. **稀疏图与空间记忆：** 由几何感知持续建立导航图，维护已探索区域并生成几何 frontier。
2. **ExploRFM 视觉评分：** 输入当前 RGB 图像和文本目标，在图像空间预测视觉 frontier、可通行性和目标相似度。公开实现采用改造的 RADIO 主干，并结合 NACLIP 与 SigLIP2 对齐。
3. **跨视角目标定位：** 粒子滤波式三角定位融合多视角目标观测，估计当前深度视野外的粗略 3D 目标位置。
4. **图节点语义评分：** 将几何 frontier 投影到图像，以视觉 frontier、可通行性、目标相似度和估计目标位置给导航图节点打分。
5. **分层规划执行：** 在语义评分后的图上选取高层目标，再由局部规划生成安全运动。

## 项目页展示的实地试验

- **开放词汇搜索：** 使用 “Orange Flag” 和 “NASA Logo” 等文本查询，展示目标相似度、目标定位与自主导航闭环。
- **Fence Approach：** 视觉评分提前发现围栏间通道；纯几何 GraphNav 在围栏进入局部地图后才绕行。页面称 WildOS 的平均距离、时间更低且波动更小，但没有给出可核查的完整数字表。
- **Dead-End Recovery：** 导航图记住已探索区域，使系统能在死胡同后回到未探索 frontier；项目页展示无空间记忆的 LRN 基线在对应演示中需要人工介入。
- **跨地形运行：** 项目页提供越野与城市环境的视频；不能据此推断覆盖了所有户外地形。

## 复现边界

官方代码仓库已公开，许可证 Apache-2.0，含 ROS 2 导航、训练与推理、图建图/规划、目标定位及部分模型检查点。运行依赖 ROS 2 Jazzy、Python 3.10+、CUDA GPU、Elevation Mapping CuPy、DLIO 和 Nav2 等外部组件。ExploRFM 的 frontier 与 traversability 头部检查点在仓库内；RADIO 主干和 SigLIP2 adaptor 需另行下载。Hugging Face 发布的是 GrandTour 图像上的视觉 frontier 标注数据，不是完整机器人多传感器日志。

## Robotics_Notebooks 映射

- 主实体：[WildOS 详情节点](../../wiki/entities/paper-wildos-open-vocabulary-object-search.md)
- 任务页：[零样本目标导航](../../wiki/tasks/zero-shot-object-navigation.md)

## 一手来源

- <https://arxiv.org/abs/2602.19308>
- <https://leggedrobotics.github.io/wildos/>
- <https://github.com/nasa-jpl/nebula2-wildos>
- <https://huggingface.co/datasets/leggedrobotics/wildos>
