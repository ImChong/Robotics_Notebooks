# 微信转载留存：EtherCAT 开源库大全（Qt历险记）

> 原文链接：https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ ；版权归原作者，仅留存条目线索供核查。


EtherCAT 开源库大全（主站 / 从站 / 辅助组件，带Github地址+选型，适合嵌入式/上位机）
分类：EtherCAT主站、EtherCAT从站、语言绑定/封装、伺服/CIa402配套、仿真工具
✅ 一、主站 Master（最常用）
1. SOEM（Simple Open EtherCAT Master）【首选入门、跨平台】
语言：C，用户态
仓库：https://github.com/OpenEtherCATsociety/SOEM
协议：CoE, FoE, EoE, Distributed Clock DC
特点：跨平台 Linux/Windows/RTOS，无内核模块，移植简单；ROS/机器人领域大量使用；v2.0大幅优化DC时钟；GPLv2+链接例外（商用友好）
适用：上位机、嵌入式Linux、STM32+H7等嵌入式平台，快速验证EtherCAT网络
2. IgH EtherCAT Master（EtherLab）【高性能硬实时】
语言：C，Linux内核模块
仓库：https://gitlab.com/etherlab.org/ethercat
特点：内核态，实时性最强，DC同步精度极高；工业CNC、机器人大量生产使用；仅支持Linux(Preempt-RT/Xenomai)，移植难度高
适用：需要微秒级同步，Linux实时工控主机
3. KickCAT
语言：C++
仓库：https://github.com/leducp/KickCAT
特点：C++面向对象，自带软件模拟从站，适合上位机开发调试；项目活跃度一般
4. Ethercrab
语言：Rust
仓库：https://github.com/ethercrab-rs/ethercrab
特点：纯Rust安全实现，无unsafe依赖，适合Rust嵌入式开发，相对年轻
5. CherryECAT
语言：C，基于IgH改造
仓库：RT-Thread社区
特点：专为MCU RTOS（FreeRTOS/RT-Thread）裁剪，轻量化
6. libethercat（DLR德国宇航中心）
• C库，机器人领域，用于空间机电控制，偏科研
✅ 二、从站 Slave 开源协议栈
1. SOES（Simple Open EtherCAT Slave）配套SOEM同组织
仓库：https://github.com/OpenEtherCATsociety/SOES
C语言，轻量从站栈，适配ESC芯片：ET1100, LAN9252, ET1815；STM32+LAN9252非常常用，做自定义EtherCAT从站（传感器、IO板）
2. Beckhoff SSC（EtherCAT Slave Stack Code）
• 官方从站栈，ETG会员免费获取源码，工业标准，功能最全；需要搭配ESC芯片，商用项目首选。非Github公开仓库，ETG官网申请。
✅ 三、配套封装 / 语言绑定（上位机Qt/Python）
1. pysoem：Python绑定SOEM，快速脚本调试 EtherCAT，适合上位机快速验证
2. soem-qt：Qt C++封装SOEM，上位机Qt项目直接调用（你做Qt上位机重点看这个）
3. ROS2-ethercat：ROS2封装SOEM / IgH，机器人关节控制
✅ 四、伺服CiA402运动控制配套库（EtherCAT伺服驱动器）
1. ecat_servo：CiA402伺服状态机实现
仓库：https://github.com/kubabuda/ecat_servo
2. OpenSML：SoftMotion 轻量运动控制库，适配CiA402，点位、插补
仓库：https://github.com/feecat/OpenSML
✅ 五、仿真/调试工具
1. EtherCAT Master Sim：软件虚拟从站，不需要真实硬件，电脑直接调试主站代码（KickCAT自带）
2. ethercat-master-sim：虚拟EtherCAT从站，用于PC端开发测试
📌 快速选型一句话
Qt上位机、Windows/嵌入式Linux，快速开发：SOEM
Linux硬实时，机器人/CNC高同步：IgH EtherLab
自己做硬件从站（薄膜传感器IO板）：SOES + LAN9252
Rust项目：Ethercrab
避坑提醒
SOEM用户态，DC时钟精度弱于IgH；高同步伺服场景优先IgH
EtherCAT协议本身不开源，协议栈代码开源，ESC硬件芯片需要单独购买
SOEM在Windows下普通网卡可用，但实时性能差，Windows做硬实时建议搭配RTOS扩展
          