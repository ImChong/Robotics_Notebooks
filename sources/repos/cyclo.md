# cyclo（ROBOTIS Cyclo Physical AI 框架索引仓）

> 来源归档

- **标题：** Cyclo — Unified Physical AI framework
- **类型：** repo（框架索引 / 模块导航，非单体应用）
- **链接：** https://github.com/ROBOTIS-GIT/cyclo
- **机构：** 乐百机器人（ROBOTIS）
- **Stars：** ~39（2026-08）
- **许可：** Apache-2.0（README 徽章）
- **入库日期：** 2026-08-07
- **一句话说明：** Cyclo 公开模块导航仓：列出 Manager / Intelligence / Control / Lab 及 interfaces、applications；标明可选私有 Supervisor/Hub。
- **沉淀到 wiki：** 合并入 [robotis](../../wiki/entities/robotis.md)（不单独建 stub）

---

## 模块表（README）

| 模块 | 仓库 | 角色 |
|------|------|------|
| Cyclo Manager | `cyclo_manager` | 运维与系统管理 |
| Cyclo Intelligence | `cyclo_intelligence` | 模仿学习 / VLA 工作流 |
| Cyclo Control | `cyclo_control` | 全身控制与执行 |
| Cyclo Lab | `cyclo_lab` | 仿真与 RL |
| — | `robotis_interfaces` | 共享接口 |
| — | `robotis_applications` | 应用集成 |

Physical AI Lineup：`ai_sapiens`、`ai_worker`、`open_manipulator`、`robotis_hand`；执行器：DYNAMIXEL + DynamixelSDK。

---

## 开源状态

**部分开源（公开模块齐全）** — README 明确 private stack（Supervisor / Hub）不在公开组织内。

---

## 对 wiki 的映射

- **wiki/entities/robotis.md** — Cyclo 地图节
- 各子模块独立升格页见组织归档 [robotis-git.md](./robotis-git.md)
