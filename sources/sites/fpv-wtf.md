# fpv.wtf — WTFOS Configurator

> 来源归档

- **标题：** fpv.wtf — WTFOS Configurator
- **类型：** site（Web 配置器 / 社区门户）
- **来源：** fpv.wtf 爱好者社区
- **链接：** https://fpv.wtf/
- **版本（页面）**：v2.3.0（2026-06 抓取）
- **入库日期：** 2026-06-18
- **一句话说明：** DJI FPV 眼镜与 Air Unit 的 **wtfOS 官方 Web 配置器**：USB 连接已 root 设备后完成 Root、wtfOS 安装维护、**opkg 包管理**、启动服务、CLI 与 OSD DVR 叠加；社区包索引指向 [repo.fpv.wtf](https://repo.fpv.wtf/pigeon/)。
- **沉淀到 wiki：** [wtfos](../../wiki/entities/wtfos.md)

---

## 站点功能（首页模块）

| 模块 | 作用 |
|------|------|
| **Connect to Device** | USB 连接已上电的已 root 眼镜或 Air Unit |
| **Root** | 在受支持固件上获取 root（安装 wtfOS 前置步骤） |
| **WTFOS** | 安装、维护或卸载 wtfOS |
| **Package Manager** | 从官方仓库安装社区开发的扩展包 |
| **Startup** | 管理设备上电后自动启动的服务 |
| **CLI** | 交互式 Shell（需熟悉 ADB/设备环境） |
| **OSD Overlay** | 将录制的 OSD 叠加到 DVR 视频上 |
| **Settings** | 配置器 Web 应用行为 |
| **Wiki / Support** | 社区 Wiki 与 Discord 支持入口 |

---

## 社区与仓库

- 官方包仓库索引：[repo.fpv.wtf/pigeon](https://repo.fpv.wtf/pigeon/)
- 固件框架源码：[fpv-wtf/wtfos](https://github.com/fpv-wtf/wtfos)
- 贡献：Discord 讨论 + 向 opkg 仓库提 PR；打包与自动化构建说明见仓库文档
- 资助：[Open Collective — fpv-wtf](https://opencollective.com/fpv-wtf/donate)

---

## 与本库关系

| 资料 | 关系 |
|------|------|
| [wtfos.md](../repos/wtfos.md) | 框架 README、兼容表与开发指南 |
| [betaflight.md](../repos/betaflight.md) | 飞控 MSP OSD 经 wtfOS 包进 DJI 数字图传 |
| [betaflight-com.md](betaflight-com.md) | Betaflight 飞控配置入口对照 |
