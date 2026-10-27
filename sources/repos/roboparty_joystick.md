# roboparty_joystick（RoboParty 安卓手柄 / 控制 App）

> 来源归档

- **标题：** Roboparty Joystick — A focused Android control surface for Roboparty robots
- **类型：** repo（Android App 源码）
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/roboparty_joystick
- **入库日期：** 2026-10-27（核查快照：`main` @ `5057474`，2026-10-20 "docs: refresh joystick README and release distribution"；`versionName` 1.0.1 / `versionCode` 2）
- **一句话说明：** Kotlin + Jetpack Compose + MVI 的 Android 遥控 App：双虚拟摇杆 / 实体手柄、50 Hz UDP 与 BLE 双通道、HMAC-SHA256 挑战应答认证、两阶段 ACK 技能动作、三步确认急停、OTA 自更新。
- **沉淀到 wiki：** 是（[`wiki/entities/roboparty-joystick.md`](../../wiki/entities/roboparty-joystick.md)）
- **上级目录：** [RP1 总览仓库归档](roboparty_rp1.md)

## 开源状态

- **已开源（App 全源码）**：`app/src/main/java/com/gamepad/controller/` 下 data / domain / presentation / di 四层，约 74 个受控文件；GPL-3.0-or-later。
- **仅客户端**：机器人侧（UDP 9000 / OTA 9001 服务端）的协议实现不在本仓；协议字段只能从 README 与客户端代码反推。
- **签名密钥**：Release 由 CI 用 `KEYSTORE_BASE64` 签名，密钥不在仓库；`CryptoHelper.kt` 内预共享密钥是占位值 `your-pre-shared-secret-key` 并带 TODO，设置页“授权密码”仅为界面桩（`// TODO: 保存新密码`）。

## 关键文件清单

| 路径 | 内容 |
|------|------|
| `app/src/main/assets/connection_config.json` | 控制 / 认证 / BLE / OTA / 失效保护默认参数 |
| `data/service/{UdpControlChannel,BleControlChannel,BleConnectionManager}.kt` | UDP（664 行）与 BLE（615 + 338 行）通道 |
| `data/service/{CryptoHelper,ReliableActionSender,OtaManager,GamepadInputManager}.kt` | 签名、可靠动作、OTA、手柄输入 |
| `data/model/packet/{Control,Auth,SkillAction,Stop,Ota}Packet.kt` | 报文定义 |
| `presentation/components/*` | Compose 界面：摇杆、姿态切换、技能库、BMS / IMU / 电机面板、设置 |
| `.github/workflows/build-apk.yml`、`app/build.gradle.kts` | CI 与构建（JDK 17、compileSdk 34、minSdk 28） |

## 对 wiki 的映射

- [wiki/entities/roboparty-joystick.md](../../wiki/entities/roboparty-joystick.md)
- 关联：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[teleoperation](../../wiki/tasks/teleoperation.md)
