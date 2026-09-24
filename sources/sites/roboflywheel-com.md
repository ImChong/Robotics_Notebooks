# RoboFlywheel 官网

> 来源归档

- **标题：** RoboFlywheel · 具身数据 开放基础设施
- **类型：** site（产品门户 / 开放基础设施展示站）
- **URL：** <http://roboflywheel.com>（CDN：`https://g.alicdn.com/robo/robo/3.0.4/`）
- **联系：** RoboFlywheel@service.alibaba.com
- **GitHub 组织：** <https://github.com/RoboFlywheel-com>
- **入库日期：** 2026-09-24
- **一句话说明：** 阿里巴巴联合上海交通大学、清华大学等共建的 **具身数据开放基础设施** 门户，围绕「一个飞轮、四个开放方向」聚合 **多源数据集（LeRobot 统一格式）**、**可复现数据配方（Recipe）**、**跨引擎仿真与物理资产（Sim / World）** 与 **统一评测榜单（Benchmarks）**。

## 站点结构（2026-09-24 核查）

| 路由 | 主题 | 要点 |
|------|------|------|
| `/` | 首页 | 「一个飞轮，四个开放方向」；真机与仿真数据统一入口；覆盖家居、零售、工业、医疗 |
| `/datasets` | 数据集 | 多源数据统一到 **LeRobot V2.1**；统一 Episode Annotation Schema；ModelScope 下载入口（部分待公开） |
| `/recipes` | 数据配方 | Episode 粒度治理 → 声明式 Recipe → 一条命令产出训练 Manifest；联合发起 **数据配方开放计划** |
| `/simulation` | 仿真 | **RoboFlywheel-Sim** 跨 Isaac Sim / MuJoCo / Genesis；**EmbodiedAtoms** 412 万物理标定资产；ManiSkill3、RoboCasa365 等生态条目 |
| `/benchmarks` | 评测榜单 | 汇集 **LIBERO**、**RoboTwin-2**、**Meta-World**、**RoboCasa GR1 Tabletop** 等套件；**WB-Policy-Bench**（全身移动操作）；数据来源标注 **Evo-Studio 公开榜单** |

## 开源核查（步骤 2.5，2026-09-24）

| 资源 | 状态 | 说明 |
|------|------|------|
| 门户前端 | **已部署、闭源** | SPA 托管于阿里 CDN；站点未列前端源码仓库 |
| [RoboFlywheel-Recipe](https://github.com/RoboFlywheel-com/RoboFlywheel-Recipe) | **部分开源 / 框架待发布** | README 与 `raw2lerobot/SKILL.md` 可见；Roadmap 称 **2026-10 末** 发布核心框架与算子 |
| [RoboFlywheel-Sim](https://github.com/RoboFlywheel-com/RoboFlywheel-Sim) | **待发布** | README 写明 **Pre-release preview**，源码 **尚未公开**，计划 **2026-10 末** 开源 |
| [RoboFlywheel-Articulation](https://github.com/RoboFlywheel-com/RoboFlywheel-Articulation)（RFW-A） | **文档已开源 / 数据待发布** | RFW-A-300K（30 万铰链体 / 500+ 类）；ModelScope `RoboFlywheel/Articulation` 标 **Coming soon** |
| [RoboFlywheel-Soft](https://github.com/RoboFlywheel-com/RoboFlywheel-Soft)（RFW-S） | **文档已开源 / 数据待发布** | 10 万可仿真柔体资产；ModelScope `RoboFlywheel/Soft` 待发布 |
| [RoboFlywheel-World](https://github.com/RoboFlywheel-com/RoboFlywheel-World)（RFW-W） | **文档已开源 / 数据待发布** | 刚体资产宇宙 RFW-R；ModelScope `RoboFlywheel/Rigid` 待发布 |
| 数据集目录（/datasets） | **部分待上传** | 站点交互提示「仓库地址待补充，公开后可通过 modelscope download 获取」 |
| 评测快照（/benchmarks） | **只读聚合** | 内嵌 Evo-Studio 等公开榜单结果（`benchmarkSnapshot.js`，更新于 2026-09-21）；非独立可 fork 评测代码仓 |

**结论：** 截至入库日，RoboFlywheel 是 **开放基础设施叙事 + 部分 GitHub 文档仓 + 待发布数据/框架** 的组合；核心 Sim / Recipe 框架与大规模 ModelScope 数据集仍在 **2026-10 前后** 发布窗口。勿与 CVPR 2026 论文项目 **[RoboWheel](https://zhangyuhong01.github.io/Robowheel)**（HOI→跨本体数据引擎，HORA 数据集）混淆。

## 合作方（站点页脚 / 模块署名）

- **阿里巴巴（Alibaba）**
- **上海交通大学（SJTU）**
- **清华大学（Tsinghua）** — 榜单与跨引擎仿真模块标注「联合发布 / 联合实验室」

## 对 wiki 的映射

- [RoboFlywheel（实体页）](../../wiki/entities/roboflywheel.md)
- [RoboFlywheel GitHub 组织归档](../repos/roboflywheel-com.md)
- 交叉：[Data Flywheel](../../wiki/concepts/data-flywheel.md)、[LeRobot](../../wiki/entities/lerobot.md)、[具身数据纵深路线](../../roadmap/depth-embodied-data.md)
