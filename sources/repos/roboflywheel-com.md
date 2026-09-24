# RoboFlywheel-com（GitHub 组织）

> 来源归档

- **标题：** RoboFlywheel-com
- **类型：** repo（GitHub 组织 / 多仓生态）
- **链接：** <https://github.com/RoboFlywheel-com>
- **门户：** <http://roboflywheel.com>
- **入库日期：** 2026-09-24
- **最近复核：** 2026-09-24（项目页 + 各仓 README）
- **一句话说明：** RoboFlywheel 开放基础设施的 **GitHub 组织**，覆盖数据配方、跨引擎仿真、刚/铰/柔体物理资产四条产品线；截至入库日 **Sim 核心代码与 ModelScope 大库仍处 pre-release**。
- **代码：** <https://github.com/RoboFlywheel-com>（**部分开源**：文档与 Recipe 骨架可见；Sim / 数据集 bulk 待发布）

## 组织概况（2026-09-24）

| 仓库 | 定位 | 开源状态 |
|------|------|----------|
| [RoboFlywheel-Recipe](https://github.com/RoboFlywheel-com/RoboFlywheel-Recipe) | Episode 粒度数据治理 + 声明式 Recipe 流水线；`run(recipe=..., deployment=...)` | README + `raw2lerobot/SKILL.md` 可见；**核心框架 2026-10 末计划发布** |
| [RoboFlywheel-Sim](https://github.com/RoboFlywheel-com/RoboFlywheel-Sim) | MuJoCo / Isaac Sim / Genesis **跨引擎**场景编排、采集与策略评测 | **Pre-release**：源码未公开，**2026-10 末** 计划开源 |
| [RoboFlywheel-Articulation](https://github.com/RoboFlywheel-com/RoboFlywheel-Articulation) | **RFW-A**：Agentic + 程序化生成 **300K** 铰链体 / **500+** 类 | 论文/arXiv **Coming soon**；ModelScope `RoboFlywheel/Articulation` 待发布 |
| [RoboFlywheel-Soft](https://github.com/RoboFlywheel-com/RoboFlywheel-Soft) | **RFW-S**：可形变资产管线 + fragile-soft benchmark | **100K** 修订版资产；ModelScope `RoboFlywheel/Soft` 待发布 |
| [RoboFlywheel-World](https://github.com/RoboFlywheel-com/RoboFlywheel-World) | **RFW-W**：开放类别 3D → 仿真就绪物理资产 + 任务条件场景生成 | 统一 RFW-R/A/S 接口；ModelScope `RoboFlywheel/Rigid` 待发布 |
| [RoboFlywheel](https://github.com/RoboFlywheel-com/RoboFlywheel) | 组织 meta 仓 | 无公开 README（404） |

## Recipe 要点（RoboFlywheel-Recipe README）

- **Govern once, mix many**：Episode 为最小治理单元；昂贵治理 **只跑一次**，高频混合 **不重搬数据**。
- 输出 **Training Manifest**（唯一身份），支持 pre-training / post-training 多配方。
- 示例数据源名（站点展示）：AgiBot World、BridgeData V2、DROID、EgoDex、InternData-M1、LIBERO、RoboMIND2 等。
- LeRobot 转换 Skill：[`raw2lerobot/SKILL.md`](https://github.com/RoboFlywheel-com/RoboFlywheel-Recipe/blob/main/raw2lerobot/SKILL.md)

## Sim 要点（RoboFlywheel-Sim README）

- USD 结构化场景 + 引擎适配层；统一任务 / 录制 / 评测接口。
- 支持导出 **LeRobot** 格式轨迹；站点宣称 9 个榜单套件上三引擎成功率差 **≤ 1.9 pt**（以项目页为准，待开源后复验）。

## World 资产族（RoboFlywheel-World README）

| 分支 | 类型 | 规模（README 宣称） |
|------|------|---------------------|
| RFW-R | 刚体 | 开放类别 household 资产；URDF/MJCF/USD |
| RFW-A | 铰链体 | 300K / 500+ 类 |
| RFW-S | 柔体 | 100K 修订版 / 485 canonical types / 5 geometry families |

## 对 wiki 的映射

- [RoboFlywheel（实体页）](../../wiki/entities/roboflywheel.md)
- [RoboFlywheel 官网](../sites/roboflywheel-com.md)
- 交叉：[LeRobot](../../wiki/entities/lerobot.md)（V2.1 统一格式）、[Data Flywheel](../../wiki/concepts/data-flywheel.md)
