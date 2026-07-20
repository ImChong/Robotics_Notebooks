# midas-hand-org（GitHub 组织）

- **标题：** MIDAS Hand (@midas-hand-org)
- **类型：** repo / organization
- **链接：** <https://github.com/midas-hand-org>
- **入库日期：** 2026-07-20
- **项目页：** <https://midas-hand.com>
- **论文：** <https://arxiv.org/abs/2607.14487>
- **许可证：** 各仓库 README 为准（入库日均为公开仓库）
- **一句话说明：** UCLA MIDAS Hand **官方 GitHub 组织**：聚合 **硬件 API、MuJoCo 仿真、重定向、遥操作** 与 **通信板 PCB 文档**、项目静态站源码。

## 组织概览（截至 2026-07-20）

| 仓库 | 用途 |
|------|------|
| [midas_hand_api](https://github.com/midas-hand-org/midas_hand_api) | Python：13 路 Dynamixel + Paxini GEN3 触觉；低级通信、高级手部命令、homing/标定 |
| [midas_hand_mujoco](https://github.com/midas-hand-org/midas_hand_mujoco) | MuJoCo 全手模型（MJCF/URDF/STL）；非拇指指 PIP–DIP 闭环约束 |
| [midas_hand_retargeter](https://github.com/midas-hand-org/midas_hand_retargeter) | dex-retargeting 封装：配置、landmark 适配、中性标定、被动耦合模式 |
| [midas_hand_teleop](https://github.com/midas-hand-org/midas_hand_teleop) | MediaPipe 摄像头输入 → 重定向 → 打印/MuJoCo/真机；实时标定与调试可视化 |
| [midas_hand_communication_board](https://github.com/midas-hand-org/midas_hand_communication_board) | 掌部 TTL 配电/通信板 PCB 相关文档 |
| [midas-hand-org.github.io](https://github.com/midas-hand-org/midas-hand-org.github.io) | 项目静态站源码 |

## 快速克隆（官方 Software 页）

```bash
git clone https://github.com/midas-hand-org/midas_hand_api.git
git clone https://github.com/midas-hand-org/midas_hand_mujoco.git
git clone https://github.com/midas-hand-org/midas_hand_retargeter.git
git clone https://github.com/midas-hand-org/midas_hand_teleop.git
```

各仓库 README 含依赖、安装与示例；硬件侧另见项目页 **Parts / CAD / Assembly** 分区。

## 开源状态

- **已开源**：截至入库日，组织 **6 个公开仓库**，覆盖控制/仿真/重定向/遥操作/PCB 文档与站点源码；CAD/BOM/装配在 **midas-hand.com**（Onshape + 下载区）而非单一 monorepo。
- **未在 GitHub 集中发布**：完整 Onshape 可编辑 CAD 需从项目页 **CAD** 区进入；触觉/电机套件为第三方商业件（Paxini、Robotis）。

## 对 wiki 的映射

- [MIDAS Hand](../../wiki/entities/midas-hand.md) — 软件栈与工程实践
- [灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md) — MediaPipe / MANUS 双模态遥操作参考
