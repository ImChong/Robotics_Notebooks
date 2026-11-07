# Gaussian-SLAM — 原论文深读

- **论文：** https://arxiv.org/html/2312.10070（arXiv:2312.10070v2）
- **官方代码：** https://github.com/VladimirYugay/Gaussian-SLAM
- **通读：** 已完成
- **核对范围：** 阅读方法 §3、实测 §4、附录实现/消融、结论及 README+真实源码调用入口；未独立运行。

## 方法

主循环对每帧 RGB-D 先用已有子地图可微渲染相机视角的 color/depth，再结合光度和几何损失直接优化相机位姿；Mapper 用透明度与深度误差构造新高斯 seeding mask，对已知区域做稀疏采样更新。地图分为独立 submaps，只优化活跃子图，其他子图可卸载；区分场景大小和地图局部开销。论文 §3.2/3.3/3.4 均有明确算法。

## 结果

原论文 arXiv:2312.10070v2 Table2 TUM-RGBD 八个序列 train-view RGB 渲染 PSNR 平均 **42.08 dB**、SSIM **0.996**、LPIPS **0.018**。**这是训练视角的 PSNR，不是未知相机视角的新视图指标**；与 SplaTAM/MonoGS 的其它 split 不应直接同列。作者 README 声称随机种子 0/1/2 三次结果取平均，Gaussian rasterizer 仍可能非确定性。

## 局限

分块子地图存在跨边界地图一致性和全局回环的工程挑战；深度质量、初始位姿和 RGB-D 数据同步会改变优化；GPU 上 Gaussian rasterizer 非完全确定，必须报告重复 seed 的波动。未经测试不可把论文渲染质量解释为机器人在任意低纹理环境里成功定位。

[独立研究节点](../../wiki/entities/paper-gaussian-slam.md)
