# NVIDIA cuRoboV2-Fast 官方项目页归档

> 来源归档（site）；2026-11-08 已再次核实 NVIDIA 原始项目主页。

- **标题：** cuRoboV2-Fast: Fast Torque-Limited Collision-Free Motion Planning
- **类型：** site
- **链接：** <https://research.nvidia.com/labs/lpr/curobov2-fast/>
- **官方论文 PDF：** <https://research.nvidia.com/labs/lpr/curobov2-fast/paper.pdf>
- **arXiv：** <https://arxiv.org/abs/2611.06913>
- **代码：** <https://github.com/NVlabs/curobo>
- **文档：** <https://nvlabs.github.io/curobo/latest/>
- **核对范围：** 项目页 Abstract、Approach、Planning/Compute、GPU kernels、Pick-and-Place、Real Robot、视频演示；与官方论文和 GitHub 仓库核对。
- **通读：** 已完成
- **入库日期：** 2026-11-06；复核：2026-11-08
- **沉淀到 wiki：** [cuRoboV2-Fast 主节点](../../wiki/entities/paper-curobov2-fast.md)

## 项目页核验

- **开源状态：已开源。** NVIDIA 页面公布 NVlabs/curobo 的实现，项目论文指明 Apache-2.0；不将论文所用的外部感知、抓取系统误称为同一仓库全部开放。
- 3 kg payload 的 RTX 5090 平均规划时延 4.4 ms（精确表值 4.43 ms），比 cuRoboV2 快 14.7×；Jetson AGX Orin 25.99 ms，较该平台上的 cuRoboV2 快 23.2×。
- 双目相机 YAM 真机使用 RTX 6000 Ada，抓放完成周期平均 5.93 s，较 cuRoboV2 缩短 34.6%；论文 Fig. 19 的规划占总周期约 1.0%。
- 方法由 IK / 按需 GPU BiRRT / B-spline 轨迹优化 / TimeOpt / 执行可行性验证构成；可微逆动力学与动态负载参数在优化中约束关节力矩。
- 原图和性能数据应引用 NVIDIA 官方页与论文，避免从二手短视频推导跨设备性能。

## 一手来源关联

- [论文原文归档](../papers/curobov2_fast_arxiv_2611_06913.md)
- [源码归档](../repos/curobov2_fast.md)
