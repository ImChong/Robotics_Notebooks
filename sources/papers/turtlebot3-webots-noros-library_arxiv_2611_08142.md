# Educational robotics beyond ROS：TurtleBot3 的 Webots 控制器直接迁移库（arXiv:2611.08142）

> 来源归档（paper · 原文通读）

- **标题：** Educational robotics beyond ROS: a custom library for direct transfer of Webots controllers in the Turtlebot3
- **类型：** paper
- **作者：** Jaime P. Pérez-Moro、Alejandro Romero、Hector Quintian、Francisco Bellas
- **单位：** 拉科鲁尼亚大学（Universidade da Coruña, UDC）CITIC 研究中心，GII 与 CTC 两个研究组（论文首页）
- **arXiv：** <https://arxiv.org/abs/2611.08142>（v1，2026-11-05，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.08142v1>（首页标题与编号已核对一致；共 9 页正文加参考文献）
- **代码：** <https://github.com/jpperezm/TurtleBot3-NoROS-Library>（论文脚注 3 给出；本次未能访问仓库，README 未核实）
- **演示视频：** <https://youtu.be/bzEeayc3Oj4>（论文脚注 4；未观看）
- **入库日期：** 2026-11-09
- **一句话说明：** 用 Python 库复刻 Webots 的设备 API，使同一份 Webots 控制器无需 ROS 即可在 TurtleBot3 Burger 真机上运行；在 6 名学生的硕士课程中使用并收集问卷反馈。
- **核对范围：** 通读全文 pdftotext 文本（摘要、§1 引言、§2.1–2.3 库设计与操作验证、§3 结果、§4 结论、参考文献 1–12），并查看 Fig. 4 与 Table 1 页面渲染图。Fig. 1 连接图、Fig. 2 执行流程图、Fig. 3 场景图只看了正文说明，未逐像素核对；论文无附录或补充材料。GitHub 仓库与 YouTube 视频未访问，库的实际实现与 README 未核对。
- **通读：** 已完成

## 开源状态

- 论文称库、部署脚本与文档在 GitHub 公开（§2.2 脚注 3）；仓库内容本次未核实。

## 核心摘录（均出自论文）

1. **动机（摘要、§1）：** ROS 已是高校机器人课程标配，但安装、配置与分布式编程模型对计算机背景弱的 AI 方向硕士生是入门门槛；论文原话称在许多此类课程中安装 Linux 或虚拟机因课时与机构支持而困难。TurtleBot3 Burger 在该校是「ROS 专用平台」。
2. **核心设计（§2.2）：** 库的关键设计决策是尽可能镜像 Webots API，函数名、行为与返回值与 Webots 对应项一致，面向对象架构，每个部件一个类：`Robot`（入口，提供 `step(timestep)` 模拟 Webots 时间步）、`Motor`、`PositionSensor`、`Accelerometer`、`Gyro`、`InertialUnit`、`Lidar`。例如把编码器计数转成弧度，与 Webots 一致。控制器开头一小段代码检测 Webots 的 `Robot` 类是否可用，不可用（即在真机上）就改为导入自定义库（Fig. 2）。
3. **硬件通路（§2.2）：** Raspberry Pi 3 主机；两个 Dynamixel XL430-W250 轮电机；OpenCR 1.0 板（带 IMU），出厂固件面向 ROS，被用 Arduino IDE 与 Dynamixel2Arduino 重新编程，主循环只响应两类串口请求（设轮速、读编码器），IMU 额外暴露原始加速度计、陀螺仪与解算的 roll、pitch、yaw；LDS-01 激光雷达经独立 USB2LDS 板，用单独线程读串口。树莓派侧用 pyserial。
4. **部署脚本（§2.2）：** 基于 `scp` 与 `ssh`，Linux 与 Windows 都能用，三条命令 `deploy`（传控制器到给定 IP 的机器人）、`run`（传输并执行）、`clean`（删除已传文件）。
5. **软件栈（§2.2）：** 学生电脑 Webots R2025a（Windows 或 Linux）；树莓派 Ubuntu Server 22.04 LTS、Python 3.10、pyserial 3.5；Arduino IDE 2.3、Dynamixel2Arduino 0.8.1。
6. **操作验证（§2.3）：** 分项各做一次：用编码器前进 25 cm、用陀螺仪左转 90 度、前进到 LiDAR 目标读数。论文只说「仿真与真机接近且高度可重复」，并明说目的不是刻画传感器误差；**未给出任何数值误差或重复次数**。综合实验：前进 25 cm（编码器）、左转 45 度（陀螺仪）、边前进边用 LiDAR 避障，20 cm 内检测到障碍则转 90 度；先在 Webots 调好，实验室按仿真场景布置，同一份代码经脚本传到真机「无任何修改」，称两端「可靠地复现了编程序列」。
7. **教学实验（§3）：** 2025/26 学年「Introductory Autonomous Vehicles」，硕士「Industrial Informatics and Robotics」，**共 6 名学生**（该课非选修）。任务：设计一个慎思式（deliberative）自主架构，完成建图、路径规划、导航三个子问题，仿真与真机都要做。场地为网格，最大 6×10 格、每格 40×40 cm，教师给 3 个仿真环境并在现实搭建其中一个。定位先用 Webots supervisor 的全局定位便于调试，最终必须改为只用编码器与陀螺仪的里程计；建图为二值占据栅格，LiDAR 在每格中心停车后读取。机器人初始位姿不全局估计，以自身初始朝向定义地图系。
8. **学生反馈（§3，Table 1，Fig. 4）：** 期末问卷。Table 1 为开放问题「相对 ROS，直接用纯 Python 编程有什么主要优势」的 5 条回答（更简单、聚焦控制逻辑，不必考虑节点分布与通信、包 / 服务 / 消息 / 发布订阅，便于接入 NumPy、优先队列等库）；表注说明这批学生因另一门课的理论课有基础 ROS 知识。Fig. 4 为题干「在真机上测试代码让我了解了仅靠 Webots 仿真学不到的现实问题（摩擦、传感器不精确等）」的评分：平均 4.83，Level 5 为 5 票，Level 4 为 1 票（据图读出，共 6 票）。
9. **结论与局限（§4）：** 作者明说目的不是取代 ROS 作为标准中间件，而是为以自主机器人本身而非中间件配置为重点的课程提供更简单的教学路径。库目前只覆盖课程所需设备：轮电机与编码器、IMU、LiDAR；新增经串口 / USB 的设备只需新增一个镜像 Webots 节点的类；RGB-D 相机等高带宽传感器需要不同的采集路径，留作未来工作。

## 与 Webots 选型相关的说明（§2.1）

Webots 被选中是因为：开源、ODE 物理引擎、控制器可用 Python / C / C++ / Java / MATLAB、内置 TurtleBot3 Burger 模型、单一界面、文档与社区。运动仿真精度与其他仿真器的比较，论文只引用 Farley 等人 [10]，自身未做比较。

## 对 wiki 的映射

- [paper-turtlebot3-webots-noros-library](../../wiki/entities/paper-turtlebot3-webots-noros-library.md)
