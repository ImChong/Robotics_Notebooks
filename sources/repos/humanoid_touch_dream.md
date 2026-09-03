# humanoid-touch-dream — 原始资料归档

- **来源**：https://github.com/chrisyrniu/humanoid-touch-dream
- **类型**：repo
- **机构**：卡内基梅隆大学 / 德克萨斯大学阿灵顿分校 / 博世人工智能中心
- **归档日期**：2026-08-26
- **Stars**：约 87（2026-08-26）
- **许可：** MIT
- **会议：** IROS 2026
- **一句话说明：** HTD 论文官方入口仓：README + 发布清单 + `htd_wbc/isaaclab_decoupled_wbc` submodule；WBC 已发布，遥操作与 HTD 策略仍待发布。

## 为什么值得保留

- 这是 HTD 全栈的**导航入口**，避免把「策略仓」和「WBC 仓」混成一个节点
- Release checklist 把开放边界写死：已发布 vs on-going
- 项目页、论文 PDF、浏览器 WBC Demo 都从本仓 README 可达

## 仓库内容（2026-08-26）

| 路径 | 内容 |
|------|------|
| `README.md` | 论文信息、WBC 入口、发布清单、引用 |
| `htd_wbc/isaaclab_decoupled_wbc` | Git submodule → [IsaacLab-Decoupled-WBC](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC) |
| `imgs/` | teaser GIF（策略 rollout 与 WBC 跟踪） |
| `.gitmodules` | submodule 指针 |

克隆后需 `git submodule update --init --recursive` 才能拿到 WBC 源码。

## 发布清单（README，2026-08-26）

- [x] Lower-body controller：Isaac Lab 训练 + 部署
  - [x] Teacher 训练与评测
  - [x] Student 训练与评测
  - [x] Student 真机部署
  - [x] Teacher / student example checkpoints
- [ ] 全身遥操作与采数（on-going）
  - [ ] Apple Vision Pro 遥操作
  - [ ] 仿真支持与 PICO 遥操作
- [ ] HTD 策略训练与部署（on-going）

## 交叉链接

| 档案 | 关系 |
|------|------|
| [isaaclab_decoupled_wbc.md](./isaaclab_decoupled_wbc.md) | 实际可运行的 WBC 代码 |
| [humanoid-touch-dream.md](../sites/humanoid-touch-dream.md) | 项目页 |
| [humanoid_touch_dream.md](../papers/humanoid_touch_dream.md) | 论文摘录 |

## 对 wiki 的映射

- [Humanoid Touch Dream（论文实体）](../../wiki/entities/paper-humanoid-touch-dream.md)
- [HTD 解耦 WBC（实体）](../../wiki/entities/htd-decoupled-wbc.md)
- [HTD 方法页](../../wiki/methods/humanoid-transformer-touch-dreaming.md)
- [Teleoperation](../../wiki/tasks/teleoperation.md) — 采数栈仍待发布
