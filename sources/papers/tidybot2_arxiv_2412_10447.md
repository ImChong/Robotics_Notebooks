# TidyBot++: An Open-Source Holonomic Mobile Manipulator for Robot Learning

- **作者：** Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao, Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg（Princeton University; Stanford University; Dexterity）
- **链接：** https://arxiv.org/abs/2412.10447 （CoRL 2024）；项目页与官方硬件文档
- **状态：** primary_full_text_checked
- **核对范围：** 已通读 `arXiv:2412.10447` 论文主体与附录（1–540 行：powered-caster 全向底盘设计与几何推导、Phone WebXR 遥操作管线、多进程控制架构、现实公寓 6 类移动操作任务与差分底盘对照实验）及官方硬件参考文档（1–101 行）
- **通读：** 已完成

## 核心要点提取

1. **Powered-Caster 全向底盘硬件与成本设计**
   - 基于 4 组改装 **SDS MK4** 舵轮（更换 2 个 PLA 3D 打印叉臂与定制轮轴，引入偏心距 $b_x = 14\text{ mm}$ 与横向偏置 $b_y$），将常规共轴轮改造为具备被动滚转自由度的主动脚轮（Powered-Caster Holonomic Drive），从机械层面消除原地转动摩擦与轮面侧滑。
   - **Kinova 标准版规格**：外形尺寸 **50×54×37.5 cm**、底盘自重 **34 kg**（加 12 kg Kinova Gen3 臂）、额定长期负载 **60 kg**（实测加载 122 kg 配重可平稳运转）、最高行进速度 1 m/s、采用 768 Wh 便携储能供电配合热插拔 6 kg SLA 动力电池可达 **8 小时** 续航、底盘 BOM 成本约 **USD 5,400**（硬件文档另提供 `40×44×37.5 cm / 26 kg` ARX5 紧凑版与 `50×54×44.5 cm / 45 kg` Franka 重载版）。
   - **全封闭与免回零感知**：CAN 总线串联 4 个 CANcoder 绝对磁编码器（开机即知舵向零位，无需回零寻位）；实测里程计漂移小于 **1 cm/m** 与 **1°/360°**（注：受 `12.8:1` 转向减速比与 14 mm 偏心限制，舵向不可反向驱动）。

2. **WebXR 手机遥操作接口**
   - 相比传统 VR 手柄（依赖头显定位、价格高）或早期 RoboTurk 仅依赖 IMU 积分漂移严重的方案，直接利用消费级手机浏览器（iOS ARKit / Android WebXR）融合 IMU 与视觉惯性里程计（VIO）同时给出低延迟 6-DOF 手持末端姿态，并通过触摸屏滑动/双指交互切换仅移动底盘、仅动机械臂或全身协同及夹爪开合，无需额外穿戴外设或固定动捕基站。

3. **真实公寓移动操作策略实验与构型对比**
   - 采用基于 ResNet-18 视觉编码器的 **Diffusion Policy**（训练 500 epochs，每项评测 10 回合）：
     - **6 项家庭移动操作任务成功率**：打开冰箱 Open fridge（100 demos）`10/10`、丢垃圾 Take out trash（50 demos）`10/10`、擦桌面 Wipe countertop（50 demos）`9/10`、装洗碗机 Load dishwasher（50 demos）`7/10`、放衣物进洗衣机 Load laundry（50 demos）`7/10`、浇花 Water plant（50 demos）`6/10`。
     - **全向底盘 vs 差分驱动（Wipe countertop，50 demos）**：全向构型平均路径长 **2.03 m**、平均耗时 **27.4 s**、成功率 **9/10**；非完整差分驱动平均路径需绕行调整至 **4.03 m**、耗时 **65.2 s**、成功率仅 **4/10**。

## 局限性与边界条件

- 主动脚轮受齿轮比与偏心制约不可反向推拉，整套系统缺乏物理柔顺推车示教能力；此外纯里程计在光滑地面打滑长距离累计后仍需配合视觉/SLAM 外部闭环定位。
- 手机 WebXR VIO 依赖屏幕常亮与持握姿态视觉纹理，在纯白墙近距贴靠等极端单调视场下跟踪偶发短暂漂移。
