# human-humanoid-tools（hhtools）

> 来源归档

- **标题：** Human-to-Humanoid tools (hhtools)
- **类型：** repo（动作重定向与数据工作台）
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/human-humanoid-tools
- **入库日期：** 2026-07-14
- **再次核对：** 2026-11-21
- **核对范围：** 通读 `Roboparty/human-humanoid-tools` 官方 `README.md` 全部章节（4 类工作流、桌面端 0.1.6 发布形态、源码 `uv sync` 与 `hhtools web` / CLI / Agent MCP 入口、Apache-2.0 许可及 SMPL 外部模型边界）；未在本地运行桌面包或 GVHMR。
- **通读：** 已完成
- **一句话说明：** 高效开源 Human-to-Humanoid 动作重定向工具，Web 前端操作，支持 30 秒级复杂全身动作迁移、任意数据集/URDF 与机器人到机器人（R2R）互转。
- **沉淀到 wiki：** 是（`wiki/entities/human-humanoid-tools.md`）

---

## 核心能力（文内自述）

| 能力 | 说明 |
|------|------|
| Fast Retarget | Newton IK（Warp 可并行）+ Interaction-Mesh（MPC solver）；约 30s/段；批量并行 |
| Any Motion | 自动识别 BVH/GLB/SMPL；支持 AMASS、GVHMR、LAFAN1、OMOMO、PHUMA、Intermimic、Meshmimic 等 |
| Any URDF | 拖入 URDF + Mesh 自动识别，无需定制适配 |
| R2R | 机器人到机器人动作互转，跨机型动作库迁移 |
| 数据分析 | 关节轨迹、重心、接触热力图；按运动学指标筛选与导出 |

---

## 对 wiki 的映射

- [human-humanoid-tools](../../wiki/entities/human-humanoid-tools.md)
- 父级：[party-os](../../wiki/entities/party-os.md)
- 交叉：[motion-retargeting](../../wiki/concepts/motion-retargeting.md)、[newton-physics](../../wiki/entities/newton-physics.md)
