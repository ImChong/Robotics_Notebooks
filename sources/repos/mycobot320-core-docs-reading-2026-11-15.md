# myCobot 320：官方技术资料与运行入口核对

- **类型：** repo / 官方技术文档
- **核对日期：** 2026-11-15
- **核对范围：** 官方 `myCobot320-docs` 的 `gitbook-en` 分支：M5 Python 安装、示例、安全说明三个完整章节；`pymycobot/main` 的 `MyCobot320.__init__/_mesg/_res` 与继承的 `get_angles/send_angles/get_coords/send_coords`。读取范围是本页串口控制路径，不代表通读全产品 SDK。
- **通读：** 已完成
- **复现状态：** 本次为文档与源码阅读，未连接硬件、未运行控制程序，不等于实机验证。

## 主要原始资料

- [Python 安装](https://github.com/elephantrobotics/myCobot320-docs/blob/gitbook-en/10-ApplicationBasePython/10.2_320_M5-ApplicationPython/1_download.md)
- [官方 M5 运行示例](https://github.com/elephantrobotics/myCobot320-docs/blob/gitbook-en/10-ApplicationBasePython/10.2_320_M5-ApplicationPython/5_example.md)
- [安全与运动范围](https://github.com/elephantrobotics/myCobot320-docs/blob/gitbook-en/3-UserNotes/320_M5/3.1.1-SafetyInstruction/1-SafetyInstruction.md)
- [串口封装](https://github.com/elephantrobotics/pymycobot/blob/main/pymycobot/mycobot320.py)、[动作命令生成](https://github.com/elephantrobotics/pymycobot/blob/main/pymycobot/generate.py)。文档树 `8a4551b7`，SDK 树 `44d7fb1c`。

## 方法、实现与限制

M5 示例使用 `from pymycobot.mycobot320 import MyCobot320` 与 115200 波特率；安装章还残留另一路 import，因此应对齐所装 SDK 的实际模块。`send_angles` 校验六轴角度和速度，编码为协议命令，经 `_mesg` 的锁及 `_res` 写入串口；读取校验返回帧命令号，最多重试三次，失败返回 -1。`get_angles` 解码成角度；笛卡尔命令位置是毫米、姿态是角度，不能与 ROS 的米/弧度不经转换地互传。官方示例用 sleep 或 `is_in_position` 等待，发包成功不等于到位。文档还说明单关节合法不保证整体无自碰，额定载荷、固定底座及原厂电源均有约束。

## 对 wiki 的映射

- [myCobot 320](../../wiki/entities/elephantrobotics-mycobot-320.md)
