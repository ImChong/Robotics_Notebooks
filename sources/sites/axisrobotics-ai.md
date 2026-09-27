# AXIS ROBOTICS（axisrobotics.ai）

> 来源归档（ingest）

- **标题：** AXIS ROBOTICS | The Compounding Data Engine for Physical AI
- **类型：** company site
- **URL：** <https://axisrobotics.ai/>
- **技术报告（Platform）：** <https://techreport.axisrobotics.ai/>
- **AXIS 数据引擎项目页：** <https://axisaiorg.github.io/AXIS-V1/>
- **机构：** Axis Robotics（AXIS ROBOTICS）
- **入库日期：** 2026-09-27
- **一句话说明：** 去中心化、浏览器遥操作 + GPU 仿真增广 + 训练部署的 Physical AI 数据基础设施；宣称 10 万+ 贡献者与任务生成—采集—训练—优化闭环。

## 开源核查（步骤 2.5，2026-09-27）

| 资源 | 状态 | URL / 说明 |
|------|------|------------|
| GitHub 组织 **AxisAIOrg** | **部分开源** | <https://github.com/AxisAIOrg> — Web 遥操作（AxisWebInfra）、任务生成（AxisTaskGen）、轨迹清洗（AxisDataCleaning）、AXIS-V1 训练（Axis-V1-Training）等 |
| 官网 **Launch App** | **产品/平台** | 浏览器仿真采集入口；非 monorepo 全栈训练代码 |
| 博客 **Composable Library**（2026-09-25） | **叙事/实验摘要** | 无单独代码包；Expert / proxy / RSI 细节未附仓库 |
| arXiv **AXIS**（2607.21588） | **论文 + 部分开源** | 社区数据引擎与 benchmark；见 [axisaiorg-github-io](./axisaiorg-github-io.md) |

**结论：** 数据平台与 AXIS-V1 相关模块 **已部分开源**；博客所述 **Grounded RSI、轻量 proxy 选轨迹、低成本 Expert** 截至入库日 **未见** 独立公开实现仓库——以官方后续发布为准。

## 页面要点（策展）

- **瓶颈叙事：** Physical AI 缺的是 **数据多样性**，不是算力；「Compounding Data Engine」四步：Task Generation → Data Collection → Model Training → Optimization（失败挖掘 → 新任务）。
- **开放生态：** 仿真/硬件 agnostic、10 万+ 人力网络、多模态异构数据、统一 data middleware。
- **产品线（首页）：** Task Generation Engine · Sim Data Collection Platform · Mobile Egocentric App · Data-Processing Pipeline。

## 对 wiki 的映射

- [Axis Robotics 实体](../../wiki/entities/axis-robotics.md)
- [可组合能力库博客](../../wiki/entities/axis-composable-capability-library.md)
- 仓库索引：[axisaiorg.md](../repos/axisaiorg.md)
