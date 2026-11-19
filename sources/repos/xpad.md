# xpad（Linux Xbox 手柄内核驱动）

> 来源归档

- **标题：** Updated Xpad Linux Kernel Driver
- **类型：** repo（Linux kernel module / DKMS）
- **维护者：** paroj（Gustavo Padovan 等社区贡献）
- **链接：** https://github.com/paroj/xpad
- **Stars：** ~986（2026-06）
- **语言：** C
- **入库日期：** 2026-06-18
- **复核日期：** 2026-11-19
- **通读：** 已完成
- **核对范围：** 仓库 `README.md` 与单文件内核驱动源码 `xpad.c`（全文件 2617 行，含 `xpad_device[]`/`xpad_table[]` 设备匹配表、`xpad_probe`、`xpad_open`/`xpad_start_input`、`xpad_irq_in` 各协议解析、`xpad_init_ff`/`xpad_play_effect`、`xpad_led_probe` 及 GIP 协议处理）
- **一句话说明：** 面向 Xbox / Xbox 360 / Xbox 360 Wireless / Xbox One 控制器的 **Linux 内核 USB 手柄驱动** 社区维护版：在主线内核 `xpad` 之上合并更多兼容设备、需初始化的 Xbox 360 类控制器与 Guitar Hero Live 等补丁；通过 DKMS 安装后暴露 `/dev/input/js*` 摇杆设备、LED sysfs 与力反馈 event 节点，是 Linux 机器人栈里 **USB 有线 Xbox 手柄** 的底层输入层。
- **沉淀到 wiki：** 是 → [`wiki/entities/xpad.md`](../../wiki/entities/xpad.md)

---

## 支持范围与边界

| 连接方式 | 是否走 xpad | 说明 |
|----------|-------------|------|
| **USB 有线** | 是 | 本驱动主场景；DKMS 安装后加载 `xpad` 模块 |
| **蓝牙** | 否 | 配对成功后走通用 **HID** 配置，不经过 xpad |
| **Xbox One 无线适配器（WiFi）** | 否 | 需用户态 daemon，见 [medusalix/xow](https://github.com/medusalix/xow) |

覆盖手柄代际：**原始 Xbox**、**Xbox 360**（含需初始化握手的第三方兼容垫）、**Xbox One**（USB）；README 明确 **Guitar Hero Live** Xbox One 控制器有额外 poke 逻辑。

## 安装与更新（DKMS）

```bash
# 安装
sudo git clone https://github.com/paroj/xpad.git /usr/src/xpad-0.4
sudo dkms install -m xpad -v 0.4

# 更新
cd /usr/src/xpad-0.4
sudo git fetch && sudo git checkout origin/master
sudo dkms remove -m xpad -v 0.4 --all
sudo dkms install -m xpad -v 0.4
```

仓库根目录含 `dkms.conf`、`Makefile`、`xpad.c`（单文件驱动实现）。

## 用户态可见接口

每个已连接手柄通常对应三类节点：

1. **`/dev/input/jsN`** — 经典 Linux joystick API（`jstest /dev/input/js0`）
2. **`/sys/class/leds/xpadN/brightness`** — 环形 LED 模式控制（0–15 种模式）
3. **`/dev/input/event*`** — 通用 input 子系统 + 力反馈（`fftest /dev/input/by-id/usb-*360*event*`）

上层 ROS / pygame / 机器人遥操作栈一般通过 **evdev** 或 **SDL** 读取，而非直接链接内核模块。

## 与机器人工程的关系

| 场景 | 关系 |
|------|------|
| [Teleoperation](../../wiki/tasks/teleoperation.md) | Linux 工作站上 **游戏手柄遥操作**、速度指令调试的常见输入设备 |
| [Open Duck Mini Runtime](../../wiki/entities/open-duck-mini-runtime.md) | 真机部署文档使用 **Xbox One 蓝牙** 配对；USB 有线场景可依赖 xpad |
| [RIO](../../wiki/entities/robot-io-rio.md) | 跨形态 I/O 框架支持手柄类遥操作入口 |
| MuJoCo / Pygame 手柄仿真 | 如 [WalkerE3 手柄仿真](../../wiki/entities/jackhan-mujoco-walke3-simulation.md) 依赖 OS 层稳定 joystick 设备 |

**常见坑：** 蓝牙与 USB 是两条路径；第三方 Xbox 360 兼容垫有时需本仓库相对主线更早的初始化补丁；调试可用 `dmesg --level=debug --follow` 配合 `jstest` 抓包。

## 对 wiki 的映射

- 新建 **`wiki/entities/xpad.md`**：Linux 机器人栈中的 Xbox USB 手柄驱动实体页
- 交叉更新：`wiki/tasks/teleoperation.md`（Linux 手柄基础设施）、`wiki/entities/open-duck-mini-runtime.md`（Xbox 输入设备对照）

## 外部参考

- [paroj/xpad（GitHub）](https://github.com/paroj/xpad)
- [medusalix/xow](https://github.com/medusalix/xow) — Xbox One 无线适配器用户态方案
- Linux `Documentation/input/input.rst` — input 子系统与 evdev 概述

---

## `xpad.c` 源码深读（2026-11-19）

- **驱动注册与设备匹配：** 通过 `module_usb_driver(xpad_driver)` 向 Linux USB 核心注册；静态表 `xpad_device[]` 枚举 VID/PID、协议类型 `xtype`（`XTYPE_XBOX`、`XTYPE_XBOX360`、`XTYPE_XBOX360W`、`XTYPE_XBOXONE`）、映射特性 `mapping` 及 Quirks（如 `QUIRK_GHL_XBOXONE`），`xpad_table[]` 供自动模块加载使用。
- **设备探测与输入/力反馈/LED 注册（`xpad_probe`）：** 分配 `struct usb_xpad`、DMA 一致性缓冲 `idata` 与输入中断 URB `irq_in`；依次调用 `xpad_init_output()`（建立异步输出队列与 `irq_out` URB，支持 rumble 与握手命令）、`xpad_init_input()`（配置 ABS 轴 / KEY 按键并注册 `/dev/input/event*` 与兼容性 `js*`）、`xpad_init_ff()`（基于 `input_ff_create_memless(..., xpad_play_effect)` 暴露 `EV_FF` / `FF_RUMBLE` 无内存力反馈效应）以及 `xpad_led_probe()`（创建 `/sys/class/leds/xpad%d`）。
- **URB 中断回调与多协议解包（`xpad_open` → `xpad_start_input` → `xpad_irq_in`）：**
  - 普通有线手柄在 `xpad_open()` 时提交 `usb_submit_urb(xpad->irq_in, GFP_KERNEL)`；`XTYPE_XBOX360W` 无线接收器则在 `xpad_probe()` 中直接调用 `xpad360w_start_input()` 监听连接包并通过 `xpad_presence_work` RCU 工作队列热插拔子手柄输入设备。
  - 每次主机控制器收到中断包即触发回调 `xpad_irq_in()`，按 `xtype` 分发至 `xpad_process_packet()`、`xpad360_process_packet()`、`xpad360w_process_packet()` 或 `xpadone_process_packet()`，通过 `input_report_abs()` / `input_report_key()` 与 `input_sync(dev)` 上报 Linux Input 子系统后重新提交 URB。
  - Xbox One 走 **GIP（Game Input Protocol）**：输入包 `GIP_CMD_INPUT (0x20)`、电源握手 `GIP_CMD_POWER (0x05)`、Xbox 引导键 `GIP_CMD_VIRTUAL_KEY (0x07)`、震动指令 `GIP_CMD_RUMBLE (0x09)`；Guitar Hero Live Xbox One 控制器另由 8 秒周期定时器 `ghl_poke_timer` 保活。
