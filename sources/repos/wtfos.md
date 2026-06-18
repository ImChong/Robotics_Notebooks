# wtfOS (fpv-wtf/wtfos)

> 来源归档

- **标题：** wtfOS
- **类型：** repo（DJI 数字 FPV 固件改造框架）
- **来源：** fpv.wtf 社区
- **链接：** https://github.com/fpv-wtf/wtfos
- **配置器：** https://fpv.wtf/
- **Stars：** ~323（2026-06）
- **入库日期：** 2026-06-18
- **许可证：** MIT
- **一句话说明：** 面向 **DJI FPV 眼镜与 Air Unit** 的社区固件改造框架：在 [margerine](https://github.com/fpv-wtf/margerine) root 之上提供 **防砖备份分区、opkg 包管理、dinit 服务管理、厂商服务 modloader** 与 Web 配置器，可安装 MSP OSD 等社区包——与 [Betaflight](betaflight.md) 飞控层、PX4 自主栈 **不同层级**（图传/显示端而非飞控）。
- **沉淀到 wiki：** [wtfos](../../wiki/entities/wtfos.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位（README）

- **赛道**：DJI **数字 HD FPV** 眼镜（Goggles V1/V2）与 **Air Unit / Vista** 图传端的社区扩展，而非飞控姿态环或 MAVLink 任务栈。
- **入口**：[fpv.wtf](https://fpv.wtf/) **WTFOS Configurator**（USB 连接已 root 设备）— Root、安装/卸载 wtfOS、包管理、启动项、CLI、OSD 叠加工具。
- **依赖 root**：须先通过配置器或 [butter](https://github.com/fpv-wtf/butter) 降级到受支持固件再 root（版本因设备而异，见下）。

---

## 框架组件

| 组件 | 说明 |
|------|------|
| **wtfos-system** | 启动早期将 **系统分区副本** 挂载覆盖真分区，降低改坏固件风险；绑定向键开机可跳过（3 短蜂鸣） |
| **opkg** | 包管理；官方源 [repo.fpv.wtf/pigeon](https://repo.fpv.wtf/pigeon/)；安装前缀 `/opt/` → `/blackbox/wtfos/opt/` |
| **dinit** | 服务管理（依赖、启停）；包可安装 unit 到 `/opt/etc/dinit.d/` |
| **wtfos-modloader** | 修改厂商服务行为的扩展框架 |
| **wtfos-configurator** | Web 配置器（[fpv.wtf](https://fpv.wtf/)） |

---

## 设备兼容（README 摘录）

| 设备 | Root / wtfOS 支持要点 |
|------|------------------------|
| **DJI FPV Goggles V1** | 固件 **V01.00.0606** 或 **V01.00.0608** |
| **DJI FPV Goggles V2** | 仅 **V01.00.0606**（须 **DJI FPV 模式** 下查看真实版本；DIY 菜单可能误导） |
| **DJI Air Unit** | 0606 / 0608 |
| **Air Unit Lite**（Caddx Vista / Runcam Link） | 0606 / 0608 |
| **O3 + Goggles V2** | 部分能力见社区包 [o3-multipage-osd](https://github.com/xNuclearSquirrel/o3-multipage-osd)，非完整 wtfOS 主线 |
| **O4、Goggles 2 / Integra / Goggles 3** | **暂无支持计划** |

非受支持版本需先用 **butter** 降级。

---

## 社区包与开发（README 摘录）

- 包索引：[repo.fpv.wtf/pigeon](https://repo.fpv.wtf/pigeon/)
- 代表包：[msp-osd](https://github.com/bri3d/msp-osd)（Betaflight **MSP** OSD 进 DJI 数字图传）、[dfbdoom](https://github.com/fpv-wtf/dfbdoom)、[djifpv_enable_live_audio](https://github.com/funneld/djifpv_enable_live_audio)
- 原生开发：Android **NDK**，target **android-23**，ABI **armeabi-v7a**
- 打包指南：[D3VL developing wtfos packages](https://d3vl.com/blog/developing-wtfos-packages/)；示例 [ipk-example](https://github.com/stylesuxx/ipk-example)

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [fpv-wtf.md](../sites/fpv-wtf.md) | 配置器 Web 应用与功能入口 |
| [betaflight.md](betaflight.md) | 飞控 **MSP OSD** 经 msp-osd 等包叠加到 DJI 图传 |
| [px4_autopilot.md](px4_autopilot.md) | 自主飞控栈；图传改造 **不替代** PX4 |
| [multirotor_uav_stack_catalog.md](multirotor_uav_stack_catalog.md) | 多旋翼栈索引；本仓补 **数字图传/显示端** 维度 |

---

## 对 wiki 的映射

- 新建实体页 [`wiki/entities/wtfos.md`](../../wiki/entities/wtfos.md)：DJI 数字 FPV 固件改造框架、与 Betaflight/PX4 分层、设备兼容与包生态。
- 更新 [`wiki/overview/multirotor-simulation-planning-control-stack.md`](../../wiki/overview/multirotor-simulation-planning-control-stack.md)：FPV 链路增加 **图传/眼镜端** 说明。
- 交叉 [`wiki/entities/betaflight.md`](../../wiki/entities/betaflight.md)：MSP OSD 与 wtfOS 包生态衔接。
