# RT-Thread

> 来源归档

- **标题：** RT-Thread — 开源物联网实时操作系统
- **类型：** repo
- **组织：** [RT-Thread](https://github.com/RT-Thread)
- **链接：** <https://github.com/RT-Thread/rt-thread>
- **项目文档：** <https://rt-thread.github.io/rt-thread/>
- **官网：** <https://www.rt-thread.io/>
- **许可证：** 当前主仓库 Apache-2.0；早期版本的许可边界应以对应 tag 为准
- **Stars：** 约 12.3k（2026-10-04）
- **入库日期：** 2026-10-04
- **一句话说明：** 面向 MCU 与 IoT 设备的开源 RTOS，提供实时内核、板级支持包、组件和软件包生态，可作为机器人嵌入式控制与设备侧软件的候选底座。

## 项目定位

RT-Thread 以 C 实现，包含线程调度、同步原语、内存管理和定时器等内核能力，并提供设备框架、文件系统、网络组件和 FinSH 等服务层。Standard 与 Nano 版本面向不同资源级别的 MCU；仓库列出 ARM、RISC-V、MIPS、x86 等架构与多种 BSP。

## 机器人系统中的使用边界

RT-Thread 可运行于嵌入式 MCU / 设备端，承接传感器、执行器、通信和看门狗等实时任务。它与 Linux + ROS 2 主控属于不同层级；是否直接承载高自由度人形机器人的主运动策略，应结合算力、驱动、实时性、通信接口与生态验证，不能只因它是 RTOS 就推断适合替换主控系统。

## 对 wiki 的映射

- [RTOS 与实时调度](../../wiki/concepts/rtos-realtime-scheduling.md)
- [项目文档归档](../sites/rt-thread-github-io.md)
