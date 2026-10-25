# HX Friends（ROBOTIS）

> 来源归档

- **标题：** HX Friends — ROBOTIS 开源机械手参考设计集
- **类型：** repo / 官方产品文档与 URDF 资源
- **机构：** 乐百机器人（ROBOTIS）
- **产品文档：** <https://docs.robotis.com/docs/systems/hx_friends/>
- **URDF / 网格源码：** <https://github.com/ROBOTIS-GIT/docs/tree/main/docusaurus/static/assets/hx>
- **控制与 ROS 包：** <https://github.com/ROBOTIS-GIT/robotis_hand>
- **入库日期：** 2026-10-25
- **沉淀到 wiki：** 是 → [robotis-hx-friends](../../wiki/entities/robotis-hx-friends.md)

---

## 项目定位

HX Friends 是建立在 DYNAMIXEL 执行器生态上的模块化手部设计集合。官方页面提供不同自由度的手、夹爪与单指设计，用于比较执行器选择、关节堆叠和操作能力，并鼓励基于模块继续改造。屏幕截图里展示了五种方案及模块化、触觉感知和 ROS 2 等介绍；下表以官方产品页与固定版本的文档仓库为准。

## 型号与模型可用性

| 型号 | 类型 / 自由度 | 官方资料中的结构 | 可解析 URDF / 网格 |
|------|----------------|------------------|--------------------|
| HX1-D03 | 单指，3 DoF | 三个全俯仰关节、指尖触觉传感器 | 有，官方 docs 仓库 |
| HX2-D01 | 夹爪，1 DoF | XM430 执行器驱动的夹爪 | 截至入库日，官方页面写着交互模型待提供，docs 仓库未见模型文件 |
| HX3-D11 | 三指手，11 DoF | XM335 执行器、三处指尖触觉传感器 | 有，官方 docs 仓库 |
| HX4-D12 | 三指加拇指，12 DoF | XM335 执行器、四处指尖触觉传感器 | 有，官方 docs 仓库 |
| HX5-D20 | 拇指加四指，20 DoF | XM335 执行器、五处指尖触觉传感器 | 官方页面提供 URDF 下载；本库已有第三方展开版条目 |

图库可从 `ROBOTIS-GIT/docs` 固定版本加载 HX1、HX3、HX4 的 URDF 与 STL 网格。官方页面称 HX Friends 为开源设计，但这些模型未附带可识别的专属 SPDX 许可证，故图库不为新增模型猜测许可。`robotis_hand` 提供 ROS 2 控制、力/触觉接口与其他官方手部描述资源；其包许可不自动推定为文档仓中 HX Friends 资产的许可。

## 仓库关系

- `docs`：ROBOTIS 文档站源码，存放 HX1-D03、HX3-D11、HX4-D12 URDF 与网格；图库条目固定在 commit [`c88527a`](https://github.com/ROBOTIS-GIT/docs/tree/c88527af0888baea6f7ef3305bbbbb911b960759/docusaurus/static/assets/hx)。
- `robotis_hand`：ROBOTIS 手部的软件、ROS 2 驱动与描述包入口；与产品设计文档互补。
- [Robot Description Gallery Online](https://github.com/ImChong/Robot_Description_Gallery_Online)：新增 HX1-D03、HX3-D11、HX4-D12 交互模型条目。

## 一手来源

- [HX Friends 官方产品页](https://docs.robotis.com/docs/systems/hx_friends/)
- [ROBOTIS docs 中的 HX 模型资源](https://github.com/ROBOTIS-GIT/docs/tree/main/docusaurus/static/assets/hx)
- [ROBOTIS Hand 软件仓库](https://github.com/ROBOTIS-GIT/robotis_hand)
- [ROBOTIS docs 固定版本](https://github.com/ROBOTIS-GIT/docs/tree/c88527af0888baea6f7ef3305bbbbb911b960759/docusaurus/static/assets/hx)
