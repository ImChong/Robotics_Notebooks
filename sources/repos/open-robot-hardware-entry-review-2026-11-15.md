# 开源机器人硬件入口复核（2026-11-15）

- **类型：** repository / 维护复核
- **核对日期：** 2026-11-15
- **核对范围：** 下列四个官方仓库的根 README 正文，以及 Kinova `kortex_description/readme.md`；只核对文档定位、配置入口与许可，不包含全部 SDK 文档或源码，也未运行硬件。
- **通读：** 已完成

## 官方入口与核对结果

| 项目 | 本次阅读的一手资料 | 可确认的维护边界 |
|---|---|---|
| myCobot 320 | [myCobot320-docs / gitbook-en README](https://github.com/elephantrobotics/myCobot320-docs/blob/gitbook-en/README.md) | 六轴协作臂，文档分 M5 与 Pi 产品/控制器路径；Python、ROS1、ROS2 的章节须按控制器与版本选择，不能只据“支持 ROS”混用安装步骤 |
| Fourier GRX N1 | [FFTAI/fourier-grx-N1 README](https://github.com/FFTAI/fourier-grx-N1/blob/main/README.md) | 本仓是 GitHub Pages/Jekyll 的 SDK 文档站，不是 SDK 源码发行仓；README 指向 `Wiki-GRx-Deploy` 的 `FourierN1` 分支示例，文档、示例、固件与硬件物料须分别核对 |
| Kinova Gen3 | [ros_kortex / noetic-devel readme](https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/readme.md)、[kortex_description/readme](https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/kortex_description/readme.md) | 官方 ROS1 包区分 `gen3` / `gen3_lite` 及夹爪 xacro；分支须匹配 ROS 版本，不能把 Gen3 系列笼统写为单一六轴型号，也不能由 ROS1 分支推断 ROS2 接口 |
| Poppy Humanoid | [poppy-humanoid README](https://github.com/poppy-project/poppy-humanoid/blob/master/README.md) | 硬件 CC BY-SA 4.0、软件 GPLv3；CAD 需从 Release 或通过 Git LFS 获取，普通网页下载可能只拿到指针；README 提供 `pip install poppy-humanoid`，不等于已验证所有当代 Python/硬件组合 |

以上为所列文档在核对日的说明，不将 README 的安装指令或历史成本数字作为实机验证结果。

## 对 wiki 的映射

- [myCobot 320](../../wiki/entities/elephantrobotics-mycobot-320.md)
- [Fourier GRX N1](../../wiki/entities/fourier-grx-n1.md)
- [Kinova Gen3](../../wiki/entities/kinova-gen3.md)
- [Poppy 平台](../../wiki/entities/poppy-project-robots.md)
- [第 02 期机器人入口索引](../../wiki/overview/robot-open-source-wechat-issue02-curator.md)
