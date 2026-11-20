# Lyra 2.0 — 官方主页与开源核验

- **名称：** Lyra 2.0
- **类型：** site / project
- **项目页：** <https://research.nvidia.com/labs/sil/projects/lyra2/>
- **代码：** <https://github.com/nv-tlabs/lyra>
- **技术原文：** <https://arxiv.org/abs/2604.13036>
- **机构：** 英伟达空间智能实验室（NVIDIA Spatial Intelligence Lab）
- **核对日期：** 2026-11-20
- **核对范围：** 官方仓库 README 链接、发布内容及对应介绍
- **通读：** 已完成

## 技术定位

Lyra 2.0 区分空间遗忘与时间漂移：为每帧保留深度、相机和点云的独立缓存，不把噪声深度硬融合成全局地图；依据目标视角的可见性召回历史关键帧，通过重投影提供对应关系。训练时引入自身生成的退化历史来修正长期 drift。生成探索视频后，以 VIPE 位姿估计和 DA3 深度将视频 lift 为 3DGS/网格。

## 开放状态和复现提示

- 训练/推理代码和权重已发布，模型许可限科研开发。
- 论文的主场景是静态环境，不显式模拟动态物体；训练数据光照变化可传入生成结果，损害 3DGS 重建。代码 Apache-2.0，但模型权重使用 NVIDIA Internal Scientific Research and Development Model License，不应误称可任意商用。
- 入口与源码分层见 [归档](../repos/lyra-2.md)。

## 映射

- [项目/论文详情](../../wiki/entities/paper-lyra-2.md)
