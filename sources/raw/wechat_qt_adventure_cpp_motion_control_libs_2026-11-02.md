# C/C++ 运动控制开源库分类整理

> 原始抓取（2026-11-03，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw

C/C++ 运动控制开源库分类整理
Original Qt历险记 Qt历险记
 Nov 2, 2026, 1:55 PM
分成 轨迹生成、机器人动力学/控制、工业PLCopen运控、嵌入式电机底层、ROS生态 五大类，标注语言、协议、适用场景、许可证。
一、轨迹规划库（最常用，OTG、S曲线、Jerk限制）
Ruckig（强烈推荐）
语言：C++，附带Python绑定
许可：社区版 MPL-2.0（商用免费），Pro版闭源
能力：带jerk限制的时间最优轨迹生成，在线实时重规划、多轴PTP、支持中间路径点；替代Reflexxes
适用：机械臂、龙门、CNC、伺服上位机，微秒级计算，硬实时友好
仓库：pantor/ruckig
pyplcopen（C++17 header-only 工业运控内核）
语言：C++，Python封装
许可：MIT
能力：PLCopen标准运动控制内核，PTP、直线、圆弧、前瞻、连续插补、电子凸轮、同步、回零、轴组状态机，零分配实时周期
适用：自研运动控制器、EtherCAT上位机、替代商用运动卡内核，纯算法，不含总线驱动
FastAccelStepper
语言：C++（嵌入式）
许可：MIT
能力：步进电机S型加减速、脉冲输出，无浮点，适合AVR/ESP32/Pico
适用：单片机步进驱动、3D打印机、小型传送机构
二、机器人动力学、整机控制库（机械臂、足式、冗余机器人）
Pinocchio
语言：C++，Python绑定
许可：BSD-2
能力：刚体动力学、正逆运动学、雅克比、RNEA/ABA，机器人动力学计算核心，Crocoddyl、Stack-of-Tasks底层依赖
适用：串联机械臂、人形、四足，做动力学控制、力矩控制
Crocoddyl
语言：C++
许可：BSD-3
能力：微分动态规划DDP，最优轨迹优化、模型预测控制MPC
依赖：Pinocchio
适用：动态抓取、足式机器人、快速运动优化
Robotics Library (RL)
语言：C++
许可：BSD（商用免费）
能力：运动学+动力学+碰撞检测+路径规划+可视化，独立库，不强制ROS
适用：机械臂离线/在线规划，Windows+Linux实时内核都支持
WBC Whole-Body Control
语言：C++
许可：BSD
能力：基于QP的全身控制，任务空间控制、多任务优先级控制
适用：人形、浮动基冗余机器人、四足机器人
Control Toolbox (ct)
语言：C++
许可：BSD
能力：最优控制、MPC、自动微分、 shooting方法
适用：非线性系统、机器人、飞行器最优控制
三、工业总线 & 工业控制栈（EtherCAT、CANopen）
SOEM（Simple Open EtherCAT Master）⭐工业最常用
语言：C
许可：GPL
能力：EtherCAT主站协议栈，和伺服驱动器（汇川、埃斯顿、倍福从站）通信
搭配：Ruckig / pyplcopen 做轨迹+SOEM做总线 = 自研EtherCAT运动控制器
IgH EtherCAT Master（ethercat-master）
语言：C
许可：GPL
工业级EtherCAT主站，硬实时（Linux RT补丁），很多国产运动控制器底层
RTmotion（Intel ECI）
语言：C++
许可：开源
能力：PLCopen运动功能块，和IgH EtherCAT Master集成，适合工业自动化、AMR、机械手
SimpleFOC（Arduino-FOC）
语言：C++
许可：MIT
能力：BLDC/永磁同步电机FOC磁场定向控制，电流/速度/位置环
平台：STM32、ESP32、RP2040、Arduino
适用：嵌入式伺服、小功率无刷电机，底层电机环
四、ROS/ROS2 运动控制栈（机器人方向）
ros2_control（C++）
不是单库，是框架：HardwareInterface + Controller（joint_trajectory_controller、forward_command_controller等）
适用：ROS2机器人，把轨迹下发到硬件，和Ruckig、MoveIt配合
MoveIt2
语言：C++
许可：BSD
能力：机械臂运动规划（RRT、OMPL）、碰撞检测、轨迹插值；偏向路径规划，不是底层S曲线OTG
OMPL（Open Motion Planning Library）
语言：C++
许可：BSD
能力：采样型路径规划（RRT、PRM等），避障路径规划，不做平滑插补
五、选型快速建议
自研EtherCAT工业运动控制器（CNC/机械手）：pyplcopen（PLCopen内核） + SOEM/IgH + Ruckig（可选）
只需要多轴平滑S曲线、jerk限制轨迹：Ruckig
机械臂动力学、力矩控制、DDP优化：Pinocchio + Crocoddyl
单片机无刷FOC电机环：SimpleFOC
ROS2机械臂项目：MoveIt2 + ros2_control + Ruckig
步进电机单片机：FastAccelStepper
区分：OMPL/MoveIt是避障路径规划（去哪里）；Ruckig/pyplcopen是轨迹插补（怎么走、速度加速度jerk限制），很多项目两者搭配。



Like Author

qt · Table of Contents
Previous
Qt/C++ 人像美颜与图像处理开源库汇总
​
Qt历险记
Like
Share
Boost
Comment