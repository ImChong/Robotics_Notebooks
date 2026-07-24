# brainco_hand_service

> 来源归档

- **标题：** brainco_hand_service
- **类型：** repo
- **来源：** unitreerobotics（Unitree 官方 GitHub 组织）
- **链接：** https://github.com/unitreerobotics/brainco_hand_service
- **星标（截至 2026-07-24）：** ~16
- **最近推送：** 2026-06-01
- **主要语言：** C++
- **分类：** 灵巧手 Serial↔DDS 服务
- **入库日期：** 2026-07-24
- **一句话说明：** Brainco Revo2 灵巧手 Serial↔DDS 服务。
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree-dexterous-hand-services.md`](../../wiki/entities/unitree-dexterous-hand-services.md)
- **组织地图：** [`sources/repos/unitree.md`](unitree.md)

---

## README 要点（编译自上游）

- Each hand (left or right) is controlled by a USB-to-serial device, and each generates a pair of topics: rt/brainco/(left or right)/(cmd or state).
- The position and speed of the fingers are normalized to the [0, 1] range.
- It is recommended to set the speed of all fingers to 1.0.
- The finger indices are mapped as follows: [Thumb, Thumbaux, Index, Middle, Ring, Pinky].

## 开源状态

- **已开源**：公开 GitHub 仓库（unitreerobotics/brainco_hand_service）。


## 对 wiki 的映射

- 实体页：[`wiki/entities/unitree-dexterous-hand-services.md`](../../wiki/entities/unitree-dexterous-hand-services.md)
- 组织枢纽：[`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
