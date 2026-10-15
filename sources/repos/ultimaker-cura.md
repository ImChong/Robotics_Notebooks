# Ultimaker Cura（官方源码仓库）

- **标题：** Ultimaker Cura
- **类型：** repo
- **来源：** Ultimaker（GitHub）
- **链接：** <https://github.com/Ultimaker/Cura>
- **官网：** <https://ultimaker.com/software/ultimaker-cura>
- **许可证：** LGPL-3.0
- **Stars / Forks：** 约 7.1k / 2.2k（2026-10-15）
- **入库日期：** 2026-10-15
- **一句话说明：** 面向 FDM/FFF 3D 打印的桌面切片应用，将三维模型、打印机/材料配置与数百项切片参数转成可供打印机执行的 G-code；README 还提供社区维护的打印机配置文件入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/ultimaker-cura.md`](../../wiki/entities/ultimaker-cura.md)

---

## README 摘要

官方 README 将 Cura 定位为用于准备 3D 打印模型的切片应用，并提到数百项设置与社区维护的打印配置文件。README 提供构建、插件、机型配置、设置、翻译和第三方库等入口；可下载版本见 [GitHub Releases](https://github.com/Ultimaker/Cura/releases/latest)。

典型用法是导入模型，选择与实际设备匹配的打印机和材料，检查模型摆放并配置切片参数，预览逐层路径后导出 G-code。配置文件帮助用户起步，但不代表在不同打印机、喷嘴、材料、固件和环境下都会得到相同结果。

## 与相关仓库的边界

- **Cura 主程序：** <https://github.com/Ultimaker/Cura>，LGPL-3.0；提供桌面 UI、配置与切片工作流。
- **CuraEngine：** <https://github.com/Ultimaker/CuraEngine>，独立的 G-code 生成引擎仓库，许可证为 AGPL-3.0。不要把引擎仓库许可证与 Cura 主程序许可证混为一谈。
- **打印配置文件：** Cura README 指向社区维护的打印机配置贡献说明。使用外部配置前核对机型、喷嘴、材料和配置来源。

## 参考链接

- 官方 README：<https://github.com/Ultimaker/Cura/blob/main/README.md>
- 配置文件与设置说明：<https://github.com/Ultimaker/Cura/wiki/Profiles-&-Settings>
- 添加打印机配置：<https://github.com/Ultimaker/Cura/wiki/Adding-new-machine-profiles-to-Cura>
- 发布版本：<https://github.com/Ultimaker/Cura/releases>
