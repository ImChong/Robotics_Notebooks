# OpenBot 论文来源归档

## 元数据

- **题目：** OpenBot: Turning Smartphones into Robots
- **作者：** Matthias Müller, Vladlen Koltun
- **单位：** Intel Labs
- **发表：** IEEE International Conference on Robotics and Automation（ICRA 2021）
- **预印本：** https://arxiv.org/abs/2008.10631（v2，2021-03-10）
- **项目页：** https://www.openbot.org/（归档见 [openbot-project.md](../sites/openbot-project.md)）
- **代码：** https://github.com/ob-f/OpenBot（归档见 [openbot.md](../repos/openbot.md)）
- **开源状态：** 论文、项目页和仓库入口相互印证；仓库含硬件设计、Android/iOS/Linux 应用、Arduino/ESP32 固件与策略训练代码。仓库 LICENSE 为 MIT。
- **入库日期：** 2026-10-25

## 论文要解决的问题

研究与教育机器人常在传感、算力、通信与成本之间取舍。OpenBot 将商品手机作为移动机器人的感知、计算和通信核心，并设计约 50 美元（不含手机）的四轮差速车体，验证手机可承担跟人和室内自主导航等任务。

## 系统与方法摘录

1. **车体：** 3D 打印底盘、四个直流电机、L298N 电机驱动、Arduino Nano、轮速光学传感器与三节 18650 电池；论文单台 BOM 约 $50，多台采购估算约 $35/台。
2. **手机与控制：** Android 手机装在车体上，运行感知与驾驶策略；手机通过 USB 串口与 Arduino 通信。Arduino 输出 PWM 驱动左右电机，并回传轮速等低层传感数据。仓库后来也支持 ESP32 与蓝牙链路等不同配置，需按车体配置核对。
3. **跟人：** 量化 MobileNet SSD 检测器只保留 person 类，按置信度阈值筛选并跟踪目标，再根据检测框中心做视觉伺服。十部手机测试，现代中端手机达到约 10 FPS 或更高；最低端 Nokia 2.2 约 4–5 FPS，论文中跟随成功率为 0%。
4. **自主导航：** 以带转向命令的条件模仿学习策略为基线设计轻量驾驶网络，少于 30 分钟标注数据即可训练出有效策略。策略在手机端实时运行，使用高层“左转/右转”等路口指令，通过走廊中的路段评估。
5. **主要比较：** Xiaomi Mi9 上，论文方法在导航距离完成度为 94±5%、成功率 89±10%、碰撞 0、推理 47±2 FPS，参数量 1.3M；CIL 基线分别为 94±5%、89±10%、0、20±1 FPS，参数量 10.7M。三次试验均值。
6. **泛化：** 多手机采集的数据改善对新手机的成功率；多车体采集提升对执行器偏差与装配差异的适应性。论文仍指出策略受差速驱动和车辆执行特性影响，并非任意本体通用。

## 复现与解释边界

- 仓库 README 提供车体、固件、手机应用、控制器、Playground 和策略训练入口；训练文档给出的采集流程是手机录制手柄驾驶数据，再用 `policy` 下的 Python/TensorFlow 工具训练。
- 论文验证集中于办公走廊与受控跟人路线，实验结果不能直接外推为开放环境中的安全导航保证。
- 论文是 2021 年的设计与实验快照；仓库持续演进，当前硬件变体、依赖和通信选项可能与论文实验设置不同。

## Robotics_Notebooks 映射

- 详情节点：[OpenBot 论文实体页](../../wiki/entities/paper-openbot-smartphone-mobile-robot.md)
- 配套项目页：[OpenBot 项目页归档](../sites/openbot-project.md)
- 官方实现：[OpenBot 仓库归档](../repos/openbot.md)

## 一手来源

- arXiv 摘要与版本记录：<https://arxiv.org/abs/2008.10631>
- arXiv HTML 全文：<https://arxiv.org/html/2008.10631v2>
- 论文 PDF（仓库维护者提供）：<https://github.com/ob-f/OpenBot/blob/master/docs/paper/OpenBot.pdf>
- 官方项目页：<https://www.openbot.org/>
- 官方代码仓库：<https://github.com/ob-f/OpenBot>
