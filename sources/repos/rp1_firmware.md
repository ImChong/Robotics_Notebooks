# RP1_firmware

> 来源归档（RP1 官方仓库体系 · 固件与 BSP 入口）

- **标题：** RP1_firmware — RP1 Linux BSP、USB-CANFD 桥固件与软件包源入口
- **类型：** repo（聚合 / 子模块入口仓）
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/RP1_firmware
- **入库日期：** 2026-10-27（核查快照：`main` 的 `1cfbe77`，2026-10-08，提交说明 `Add submodules and README`）
- **一句话说明：** RP1 的固件总入口：本身只含 README、LICENSE（GPL-3.0）与 `.gitmodules`，真实内容在 `roboparty_build`（BSP）、`roboparty_EtherCANFD`（USB-CANFD 桥）、`roboparty_repo`（APT / 底层 SDK 包源）三个子仓。
- **沉淀到 wiki：** 是（[`wiki/entities/rp1-firmware.md`](../../wiki/entities/rp1-firmware.md)）
- **上级入口：** [RP1 总览仓库 README](https://github.com/Roboparty/RP1)；wiki 总览 [roboparty-rp1](../../wiki/entities/roboparty-rp1.md)

## 开源状态

- **入口仓已开源，但快照 HEAD 树不含子模块 gitlink**：`git ls-tree HEAD` 只有 `.gitmodules`、`LICENSE`、`README.md` 三项，而 README 声称“子模块固定到已提交修订”；因此 `git clone --recurse-submodules` 能否取到子仓内容未经验证（推测：需手动克隆三个子仓）。
- **子仓均公开可克隆**（2026-10-27 逐一浅克隆验证）。
- README「Deploy on the robot」链接的 `Roboparty/RP1_deploy` 在匿名访问下不可读（私有或不存在，未确认）；公开部署仓为 [roboparty_deploy](roboparty_deploy.md)。

## 子仓核查摘要

| 子仓 | 快照 | 内容 | 许可证 |
|------|------|------|--------|
| `roboparty_build` | `77101e9`（2026-10-25）；`VERSION` `26.11.0-trunk-2.0.2` | Armbian 构建框架分支；新增板型 `config/boards/robopi2.conf`（`rockchip-rk3588`，DTB `rk3588s-orangepi-cm5-robopi2.dtb`，串口 `ttyS2:1500000`） | GPL-2.0（Armbian 原文） |
| `roboparty_EtherCANFD` | `25b54a2`（2026-10-26）；固件 1.1.0 | HPM5E31（RISC-V）4 路 CAN FD ↔ USB，gs_usb 协议，VID `0x1209` / PID `0x2323`，HPM SDK 1.13.0，PowerShell + CMake/Ninja 构建 | GPL-3.0 |
| `roboparty_repo` | `87f5290`（2026-10-25） | `manifest.xml`（Android `repo` 源码清单）、`routing.yaml`（deb 路由）、`scripts/firmware-manifest.json`（工厂固件清单）、APT 签名公钥 `scripts/roboparty.gpg` | 仓库内未见 LICENSE（未核实） |

## 对 wiki 的映射

- [wiki/entities/rp1-firmware.md](../../wiki/entities/rp1-firmware.md)
- 区分：[atom01-firmware](../../wiki/entities/atom01-firmware.md)（`atom01_firmware`，提交 `f06f409`，与本仓 `1cfbe77` 不同；对应关系未核实）
- 下游：[roboparty_deploy](roboparty_deploy.md)（运行时依赖 CAN 设备与电机 / IMU 驱动）
