# A2I-Calib++（arXiv:2610.29438）

> 来源归档（paper）

- **标题：** A2I-Calib++: A Kinematics-Driven Anti-noise Active Calibration Framework for Whole-Body Multi-IMU Systems on Legged Robots
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.29438>
- **PDF：** <https://arxiv.org/pdf/2610.29438>
- **作者：** Fangyu Jiang, Chaoran Xiong, Kehui Ma, Zhen Sun, Zeyu Zhang, Ling Pei
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-10-26
- **一句话说明：** A2I-Calib++：面向腿式机器人全身多 IMU 的运动学驱动抗噪主动标定。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 日常腿式运动激励不足，而足端冲击、振动和打滑又会干扰多 IMU 时空外参及偏置标定。
3. **周报所述创新点：** 设计噪声各向同性激励轨迹，以带滚动接触奖励的 AMP 控制器执行；再利用小腿 IMU、关节运动学和接触约束构造虚拟参考 IMU，同时估计时间偏移、旋转、平移和惯性偏置。
4. **arXiv 摘要（英文节选）：** Recently, multi-node inertial measurement unit (multi-IMU) systems have shown promise for robust odometry in legged robots. Multi-IMU fusion can improve legged robot odometry but requires accurate spatiotemporal calibration among the sensors. Existing multi-IMU calibration methods, designed for aerial or handheld platforms, face challenges on legged robots because routine locomotion may provide insufficient excitation, while foot slips, impacts, and vibrations introduce measurement disturbances.

**对 wiki 的映射**

- [paper-a2i-calib-plus-multi-imu-legged-calibration](../../wiki/entities/paper-a2i-calib-plus-multi-imu-legged-calibration.md)
