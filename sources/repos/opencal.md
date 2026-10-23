# computed-axial-lithography / OpenCAL

- **标题：** OpenCAL
- **类型：** repo
- **组织：** Computed Axial Lithography（UC Berkeley）
- **源码：** <https://github.com/computed-axial-lithography/OpenCAL>
- **关联硬件仓库：** <https://github.com/computed-axial-lithography/CAL-hardware>
- **项目文档：** <https://opencal-org.readthedocs.io/en/stable/>
- **配套论文：** <https://arxiv.org/abs/2509.02865>
- **入库日期：** 2026-10-23
- **一句话说明：** 开源计算轴向光刻打印机项目，包含运行于 Raspberry Pi 5 的控制软件；配合 CAL-hardware、Tomo 与 VAMToolbox 完成机器搭建和投影打印。

## 开源状态核查（2026-10-23）

官方 GitHub 组织公开 OpenCAL 控制软件、CAL-hardware、Tomo、VAMToolbox 等仓库；OpenCAL README 标注面向研究用途。各组件采用的许可可能不同，使用或再分发前应分别查看对应仓库的 LICENSE。VAMToolbox README 特别说明其并非 OSI-approved open source，不能将整套配套软件笼统描述为同一开源许可。

## README 摘要

OpenCAL 软件面向 Raspberry Pi 5，控制旋转机构与打印流程，并通过硬件 LCD / encoder 操作。投影图像序列由 Tomo 或 VAMToolbox 生成，打印机读取生成的打印视频。

## 仓库与模块

| 资源 | 作用 |
|---|---|
| [OpenCAL](https://github.com/computed-axial-lithography/OpenCAL) | Raspberry Pi 5 打印机控制软件 |
| [CAL-hardware](https://github.com/computed-axial-lithography/CAL-hardware) | 教育与研究用途的 CAL 设备硬件设计 |
| [Tomo](https://github.com/computed-axial-lithography/tomo) | VAMToolbox 的图形界面，生成打印视频 |
| [VAMToolbox](https://github.com/computed-axial-lithography/VAMToolbox) | 体积增材制造投影计算与优化的 Python 工具箱 |

## 运行入口

主仓库 README 提供 Raspberry Pi 5 安装镜像与源码安装说明；源码运行入口为 Python 模块 `python3 -m opencal`。构建与接线步骤请遵循官方文档及硬件仓库说明。

## 参考链接

- [OpenCAL GitHub](https://github.com/computed-axial-lithography/OpenCAL)
- [CAL-hardware](https://github.com/computed-axial-lithography/CAL-hardware)
- [Tomo](https://github.com/computed-axial-lithography/tomo)
- [VAMToolbox](https://github.com/computed-axial-lithography/VAMToolbox)
- [OpenCAL 文档](https://opencal-org.readthedocs.io/en/stable/)
- [OpenCAL 论文](https://arxiv.org/abs/2509.02865)
