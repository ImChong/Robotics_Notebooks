# Robotics Bench（交互式机器人硬件学习演示）

- **类型：** 浏览器交互式硬件教学演示
- **项目页：** https://robotics-engineer.vercel.app/
- **创作工具：** [Colorfigures](https://april-zhu.com/colorfigures)
- **发布日期：** 未标注
- **入库日期：** 2026-10-23
- **源码：** 项目页未提供公开代码仓库链接；截至入库日未能确认源码开放状态
- **一句话说明：** 以可交互的等距插画展示机器人与嵌入式硬件，并通过步骤按钮演示装配、连线、通电等学习场景。
- **沉淀到 wiki：** [Robotics Bench](../../wiki/entities/robotics-bench.md)、[开源机器人学学习指南（qqfly）](../../wiki/entities/learn-robotics-qqfly-guide.md)

## 项目核查

项目页面的搜索索引标题为 “Arduino Uno R3”，页面文字包含 “Pick & place SMD”“Solder joints”等交互操作，并标注 “Made with Colorfigures”。[Colorfigures 作者页面](https://april-zhu.com/colorfigures)将其描述为生成可交互、可旋转的等距 HTML 图形的 skill；生成结果可在浏览器打开。

用户提供的截图展示了 Raspberry Pi 5 模型和按步骤进行 PCB、SMD、焊点、电源插拔等交互。画面侧栏还列出 Arduino Uno R3、ESP32-S3 DevKit、TB6612FNG、L298N、A4988、PCA9685、MPU-6050、HC-SR04 等条目。截图文案称内含 13 款模型；本次未从项目页面文本中独立核实全部 13 款。

## 开源与复现边界

截至 2026-10-23，项目页可在线访问，但未发现项目源码仓库或可复现构建说明，因此源码开放状态**未确认**。Colorfigures 是其创作工具，不应将工具本身的许可证直接推定为该演示站点的代码许可证。

页面呈现的是交互式视觉教学演示。当前可查的一手资料不足以证明它运行了电路级仿真、热仿真或可用于器件选型的真实物理模型；演示中的读数与故障效果应视作教学呈现，直到源码或技术文档能证明其仿真方法。

## 参考链接

- [Robotics Bench 在线演示](https://robotics-engineer.vercel.app/)
- [Colorfigures 官方说明与示例](https://april-zhu.com/colorfigures)
- [Arduino UNO R3 官方硬件页](https://docs.arduino.cc/hardware/uno-rev3/)
- [Raspberry Pi 5 官方产品与规格](https://www.raspberrypi.com/products/raspberry-pi-5/)

## 对 wiki 的映射

- [Robotics Bench 项目实体](../../wiki/entities/robotics-bench.md)
- [开源机器人学学习指南（qqfly）](../../wiki/entities/learn-robotics-qqfly-guide.md)
