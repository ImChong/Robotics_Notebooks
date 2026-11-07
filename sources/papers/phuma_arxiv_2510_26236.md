# PHUMA：物理可靠的人形运动数据集（通读）

> 来源归档（原文通读 · 支撑 `wiki/concepts/motion-data-quality.md`）

- **标题：** PHUMA: Physically Reliable Humanoid Locomotion Dataset
- **作者：** Kyungmin Lee, Sibeen Kim, Youngdo Lee, Minho Park, Hyunseung Kim, Dongyoon Hwang, Donghu Kim, Hojoon Lee, Jaegul Choo（KAIST / KRAFTON / Holiday Robotics）
- **类型：** paper / dataset
- **链接：** <https://arxiv.org/abs/2510.26236>（PDF：<https://arxiv.org/pdf/2510.26236>）；项目页 <https://davian-robotics.github.io/PHUMA>（本次未读）；仓库归档见 [`sources/repos/phuma.md`](../repos/phuma.md)
- **版本 / 日期：** arXiv v3，2026-10-12（CoRL 2026）
- **入库日期：** 2026-11-07
- **核对范围：** 通读摘要、§1 引言、§2 相关工作、§3 方法（物理感知筛选的抖动 / 稳定性 / 地面接触判据，PhySINK 的四项损失与五个评测指标）、§4 实验（PhySINK 消融与对比表 1/3、数据集对比表 4、真机表 5）与局限 / 结论；附录（滤波器参数、数据组成、更多机型结果）仅浏览，未逐项核对。未运行代码、未下载数据。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **数据：** 在 Humanoid-X（视频转 SMPL 再重定向，231.4 h）基础上做物理感知筛选，再并入 LaFAN1、LocoMuJoCo 与自拍视频，得到 **73.0 h / 76.0K 个 4 秒片段**，文中称是 AMASS 的 3.5 倍（约 20.9 h，由 73.0/3.5 推算），超过 Humanoid-X 231.4 h 中可行的 69.1 h；目标机器人为 Unitree G1 与 H1-2，**仅限平地 locomotion**。
- **物理感知筛选（§3.1）：** 低通 Butterworth 滤波去高频抖动；以"CoM 到支撑基（接触地面的脚所覆盖的地面区域）的距离"检测不稳定（如坐在不存在的椅子上）；用脚部网格顶点高度的多数投票估计全局地面平面，把地面对齐到 z=0，再按脚区域（脚跟 / 脚尖）计算接触得分；按 4 秒切片，丢弃（i）jerk 过大、（ii）CoM 远离支撑基、（iii）足地接触不足（悬空）的片段。
- **PhySINK（§3.2）：** 在形状自适应 IK（SINK）上加三项物理损失：关节可行性（位置 / 速度超过机械极限的 98% 即罚）、接地（接触脚高度平方）、防滑（接触脚水平速度平方）；运动保真项含全局位置、局部相邻关节位移（位置 + 方向）和二阶平滑。评测五个"满足某条件帧的百分比"指标：运动保真（关节位置误差 <10 cm 且连杆朝向误差 <10°）、关节可行（98% 极限内）、不悬空（接触时脚高于地面不超过 1 cm）、不穿透（不低于地面 1 cm）、不滑动（接触时水平速度 <10 cm/s）。对比 IK、GMR、SINK：GMR 的"不穿透"最高，但靠把整段运动降到最低接触点，导致别处严重悬空；PhySINK 是唯一在五个指标上都稳定的方法。
- **下游结果：** 在 AMASS 数据上隔离重定向影响，PhySINK 重定向的策略成功率高于 IK / GMR / SINK（提升最大的是角运动与垂直运动）。比较数据集（LaFAN1、AMASS+SINK、Humanoid-X、PHUMA）：**PHUMA 在所有运动类别与两种机器人上成功率最高**；结论是"规模与质量单独都不够"——Humanoid-X 规模大但质量差而落后，LaFAN1 / AMASS 干净但覆盖不足。真机（BeyondMimic，G1，PHUMA 测试集 24 段）：PHUMA 训练的策略比 AMASS 训练的跟踪误差低 16.3%。
- **局限（原文）：** 只覆盖平地 locomotion，不含场景交互（搬物、坐、爬）与不平地形；大量视频估计的动作在筛选中被丢弃；真机只对比了 AMASS（Humanoid-X 策略在仿真里摔倒频率约高 3 倍，硬件上有安全风险）；PHUMA 的运动**没有完全消除**物理违规。

## 对 wiki 的映射

- [`wiki/concepts/motion-data-quality.md`](../../wiki/concepts/motion-data-quality.md)

## 参考来源

- Lee, K., Kim, S. et al. (2026). PHUMA: Physically Reliable Humanoid Locomotion Dataset. CoRL 2026. <https://arxiv.org/abs/2510.26236>
