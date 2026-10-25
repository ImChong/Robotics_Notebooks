# OpenBot 官方代码仓库

> 来源归档

- **标题：** OpenBot — Turning Smartphones into Robots
- **类型：** repo
- **链接：** https://github.com/ob-f/OpenBot
- **项目页：** https://www.openbot.org/（归档：[openbot-project.md](../sites/openbot-project.md)）
- **论文：** https://arxiv.org/abs/2008.10631（归档：[openbot_arxiv_2008_10631.md](../papers/openbot_arxiv_2008_10631.md)）
- **许可：** MIT（仓库 LICENSE）
- **代码状态：** 已公开；含 `body/`、`firmware/`、`robot/`、`controller/`、`open-code/` 与 `policy/` 等实现目录。
- **核查日期：** 2026-10-25
- **一句话说明：** 从低成本轮式机器人车体、Arduino/ESP32 控制固件、手机应用到学习型驾驶策略的完整开源实现。
- **沉淀到 wiki：** [OpenBot 论文实体页](../../wiki/entities/paper-openbot-smartphone-mobile-robot.md)

## 仓库运行入口

- `body/`：车体组装与硬件版本资料。
- `firmware/`：Arduino Nano / ESP32 固件，处理串口或蓝牙命令、PWM 电机输出及车载传感器反馈。
- `robot/`：Android、iOS 与 Linux 端应用；手机端应用运行驾驶策略并与机器人车体通信。
- `controller/`：手柄、网页及其他远程控制入口。
- `policy/`：驾驶数据采集、转换、训练与评估工具。
- `open-code/`：面向应用编程的 Playground。

## 一手入口

- [仓库 README](https://github.com/ob-f/OpenBot)
- [硬件说明](https://github.com/ob-f/OpenBot/tree/master/body)
- [固件说明](https://github.com/ob-f/OpenBot/tree/master/firmware)
- [手机应用说明](https://github.com/ob-f/OpenBot/tree/master/robot)
- [驾驶策略说明](https://github.com/ob-f/OpenBot/tree/master/policy)
- [MIT License](https://github.com/ob-f/OpenBot/blob/master/LICENSE)
