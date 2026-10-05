# reactive_diffusion_policy（RDP 代码仓）

> 来源归档（ingest；依据公开 README 摘要）

- **类型：** repo / research-code
- **官方仓库：** <https://github.com/xiaoxiaoxh/reactive_diffusion_policy>
- **论文：** <https://arxiv.org/abs/2503.02881>
- **入库日期：** 2026-10-05
- **一句话说明：** RDP 官方实现，包含策略代码与自定义 task、robot、tactile/force sensor 部署指南，并链接数据与 checkpoint。

## README 所示内容

- 提供策略/实验配置、数据采集与自定义部署说明。示例硬件包括 Flexiv Rizon 4 双臂、Franka 支持、RealSense、可选 GelSight Mini 和 Quest 3 TactAR。
- 作者参考软件环境为 Ubuntu 22.04 / ROS 2 Humble、Python venv 与 PyTorch 1.13.1 + CUDA 11.7；依赖应按仓库当前说明安装。
- README 链接示例数据集、checkpoint、TactAR Unity APP、数据采集指南和触觉 embedding 指南。
- 这些仅为作者记录的部署配置，不代表唯一可行设备组合或开箱即用的最低要求。

## 获取与复现注意

- 除安装依赖外，还要配置相机/传感器、机器人网络地址、坐标标定和任务 YAML。
- 作者说明 GelSight Mini 以 24 FPS 处理触觉图像需要较强 CPU；此提示不应泛化到全部传感器。
- 本归档不代替当前仓库的 README 与 LICENSE；代码、数据及模型许可应分别核实。

## Wiki 映射

- [RDP 实体页](../../wiki/entities/paper-sa-2503-02881-reactive-diffusion-policy-slow-fast-visual-tacti.md)
- [TactAR APP 源码归档](./tactar-app.md)
- [论文来源归档](../papers/reactive_diffusion_policy_2503_02881.md)
