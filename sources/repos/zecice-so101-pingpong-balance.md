# SO-101 Ping-Pong Ball Balancing · 作者仓库核对

- **类型：** 范例项目、源码与预训练权重
- **作者：** zecice（独立维护者）
- **代码：** https://github.com/zecice/so101-pingpong-balance
- **固定版本：** [a2e6f88](https://github.com/zecice/so101-pingpong-balance/tree/a2e6f88b50e97b93e492debe4279d6047add6716)
- **作者链接的真机演示：** https://x.com/zecicc/status/2118380170662863229
- **核对范围：** 2026-11-27 通读 README、资产来源及许可；核对固定提交的 PPO 训练入口、环境随机化与成功判据、两种仿真评估、相机采集、检测、估计、速度滤波、真机控制与停止分支。核对两个权重归档以及 FdVel 的合成输入输出。真机视频在原仓库有链接，本次未独立播放、训练或测试硬件。
- **通读：** 已完成

| 公开入口 | 核对内容 |
|---|---|
| [环境与 FdVel](https://github.com/zecice/so101-pingpong-balance/blob/a2e6f88b50e97b93e492debe4279d6047add6716/sim/env.py) | 在估计器噪声模式从拍面位置差分估速度；达到时限、未提前终止且终点距中心小于 6 cm 才算成功。 |
| [PPO 评估](https://github.com/zecice/so101-pingpong-balance/blob/a2e6f88b50e97b93e492debe4279d6047add6716/sim/eval_policy.py) · [视觉评估](https://github.com/zecice/so101-pingpong-balance/blob/a2e6f88b50e97b93e492debe4279d6047add6716/perception/estimator.py) | 前者没有把图像检测接进控制观测，默认可读取检查点的状态噪声和马达随机化配置；后者在模拟图像上接入检测器、MLP 和差分。两者均在仿真。 |
| [真机运行器](https://github.com/zecice/so101-pingpong-balance/blob/a2e6f88b50e97b93e492debe4279d6047add6716/hardware/run_real.py) | 后台线程保留最新相机帧，控制循环读取关节、检测球、估状态、输出目标；超过限制保持最后目标，退出不自动卸力。 |
| [两个权重](https://github.com/zecice/so101-pingpong-balance/tree/a2e6f88b50e97b93e492debe4279d6047add6716/weights) | controller.pt 与 estimator.pt 实际存在；仅校验 ZIP 结构和 CRC，不反序列化或推理。 |

作者报告随机化仿真 97.5%、完整相机链路仿真 113/120（94.2%）；原仓库未给出 85.8% 真机率或其分母。项目代码与权重 MIT；机械臂模型 Apache-2.0，球拍网格 MIT，具体见 [资产 README](https://github.com/zecice/so101-pingpong-balance/blob/a2e6f88b50e97b93e492debe4279d6047add6716/assets/so101/README.md)。

- **沉淀到 wiki：** [SO-101 乒乓球平衡](../../wiki/entities/so101-pingpong-balance.md)
