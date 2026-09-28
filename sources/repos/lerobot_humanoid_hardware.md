# lerobot-humanoid-hardware

> 来源归档

- **标题：** lerobot-humanoid-hardware
- **类型：** repo
- **链接：** https://github.com/huggingface/lerobot-humanoid-hardware
- **许可证：** Apache-2.0
- **入库日期：** 2026-09-28
- **一句话说明：** LeRobot Humanoid **双足平台**硬件复现资产：BOM、Onshape/STL、装配与制造文档、电子接线映射、电机预组装 commissioning 工具。
- **代码：** https://github.com/huggingface/lerobot-humanoid-hardware（**已开源**）
- **CAD：** https://cad.onshape.com/documents/fb645318a27646d1d8840be6/w/d1cae8805fb652b4d1614997/e/804a1da43f242001a05129b4
- **沉淀到 wiki：** [lerobot-humanoid](../../wiki/entities/lerobot-humanoid.md)

---

## 范围（README）

- **IN：** biped platform hardware
- **OUT（当前迭代）：** upper-body hardware（Onshape 含上身模型，本仓迭代仅跟踪双足）

## 目录要点

| 路径 | 内容 |
|------|------|
| `docs/` | 架构、指南、流程 |
| `hardware/cad/` | Onshape 引用 + 导出 STL |
| `hardware/bom/` | 机器可读与人类可读 BOM |
| `hardware/config/` | 电机 commissioning（`commission_motor.py`） |
| `hardware/electronics/` | 接线与连接器映射 |

STL 组织：`hardware/cad/stl/biped_platform/{left_leg,right_leg,torso}/…`

## 推荐装配顺序（README）

1. 按 `hardware/bom/bom_buy.csv` 采购
2. 按 `docs/manufacturing/printing_guide.md` 打印 STL
3. 装配前完成电机协议/ID commissioning（`commission_motor.py wizard --channel can0`）
4. 机械子装配 → 接线与首次上电检查

**规则：** 未完成电机 commissioning 与运动检查前不要开始机械总装。

## 运行时对接

- 控制与仿真运行时：[lerobot-humanoid-runtime](lerobot_humanoid_runtime.md)（GitHub 组织页为 `huggingface/lerobot-humanoid-runtime`）

## 对 wiki 的映射

- [LeRobot Humanoid](../../wiki/entities/lerobot-humanoid.md)
