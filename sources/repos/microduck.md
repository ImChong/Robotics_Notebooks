# pollen-robotics/microduck

> 来源归档

- **标题：** Microduck
- **类型：** repo
- **组织：** pollen-robotics
- **链接：** https://github.com/pollen-robotics/microduck
- **项目页：** https://pollen-robotics.com/microduck
- **训练仓：** https://github.com/pollen-robotics/microduck_rl
- **社区：** https://discord.com/invite/pollen-community-519098054377340948
- **许可：** Apache-2.0（软件栈）
- **语言：** Rust
- **默认分支：** `main`
- **Stars / Forks：** ~1003 / 98（2026-08-28，GitHub API）
- **入库日期：** 2026-08-28
- **一句话说明：** Microduck 的机载「大脑」：Rockchip RK3566 上若干 daemon，50 Hz 用神经网络策略驱动约十五路舵机；策略在隔壁 `microduck_rl` 用 MuJoCo + PPO 训练并导出 ONNX。
- **开源状态：** **已开源**
- **项目页归档：** [pollen-robotics-microduck.md](../sites/pollen-robotics-microduck.md)
- **沉淀到 wiki：** [pollen-microduck](../../wiki/entities/pollen-microduck.md)

---

## 仓库职责

README 自称 *This repo is the duck's brain*：约 **25 cm / 800 g**，板上跑控制环、无线电、相机与防变砖的更新系统。真机购买入口是产品页，不是本仓的 BOM。

| 读者 | 入口 |
|------|------|
| 已有真机 | `docs/robot/cheatsheet.md`（`robotctl`）、手柄配对、`duckctl`（蓝牙无网）、更新/回滚 |
| 在板上开发 | `docs/design/architecture.md`、`docs/robot/install-dev.md`、`cheatsheet-dev.md`、`dev-push.md`、`CONTRIBUTING.md` |
| 训练策略 | 隔壁 [microduck_rl](https://github.com/pollen-robotics/microduck_rl) |

README 演示能力：手柄行走、装轮后切「另一颗脑子」滚动、喙触地拾取、被推倒后自起；另有坐下、踢球、前滚与叫声。

## 机载架构（`docs/design/architecture.md`，2026-07-22 draft）

七个 daemon + CLI，Unix socket 上 **JSON-RPC 2.0 / NDJSON**。**只有 `robotd` 碰电机**；`configd` / `updaterd` / `btd` 在 `robotd` 挂掉后仍须可达（配网、回滚、BLE）。

| 服务 | 职责 |
|------|------|
| `robotd` | 50 Hz 控制环、Dynamixel 总线、策略、安全；`/run/robotd.sock` |
| `configd` | Wi-Fi、身份、手柄绑定、重启 |
| `updaterd` | 验签、整目录切换 `current` symlink、健康门、失败回滚 |
| `btd` | BLE 传输适配，不持有机器人状态 |
| `padd` | 手柄 → 与 App 相同的 intent |
| `mediad` | 相机/音频、WebRTC；控制台 `:8080`、信令 `:8443` |
| `tofd` | 头上 8×8 ToF，只发布不读他人 |
| `robotctl` / `duckctl` | CLI；后者走蓝牙、不依赖 ssh |

发布模型：**整包替换而非热补丁**。构建落入 `/opt/robot/daemon/releases/`，`updaterd` 验签后改 symlink、重启 unit，再问 `robot.health`；不健康则自行切回旧版。

顶层 crate 还包括 `kinematics`、`odometry`、`policies`、`deploy`、`duck-ipc-proto` 等。

## 电机数量口径

- 产品页 / Runtime README：**15 motors / fifteen servos**（与 IMU 板共用一条串口）。
- RL 仓任务图：**14 路 Dynamixel XL330**（左右腿各 5 + 颈/头 4）进入 MJCF 执行器。
- 入库写法：硬件规格以产品页 15 路为准；策略观测/动作为 14 路伺服关节。第 15 路不在 RL 关节表里，勿把两边数字直接等同。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 整机与 Runtime | `wiki/entities/pollen-microduck.md` |
| 训练与 sim2real 菜谱 | `wiki/entities/pollen-microduck-rl.md` |
| 同公司 Reachy2 | `wiki/entities/pollen-reachy2.md` |
