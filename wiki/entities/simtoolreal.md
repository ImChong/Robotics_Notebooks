---
type: entity
tags:
- loco-manipulation
- humanoid
- contact
- hmi-opensource-table
- repo
- linux-foundation
- paper
- awesome-real2sim2real
- sun254667-r2s2r
status: draft
updated: 2026-10-06
summary: SimToolReal：程序化工具随机化几何和动力学，KUKA iiwa14与22自由度SHARPA手跟踪统一6D目标轨迹。140维状态输入和SAPG训练可用于研究跨工具Sim2Real，真实闭环仍依赖外部视觉与机械臂控制仓库。
related:
- ../tasks/loco-manipulation.md
- ../concepts/whole-body-control.md
- ../entities/humanoid-motion-intelligence.md
- ../queries/hmi-opensource-projects-coverage.md
- ../entities/awesome-real2sim2real.md
- ../overview/sun-awesome-r2s2r-technology-map.md
- ../methods/reinforcement-learning.md
- ../methods/crisp-real2sim.md
- ../tasks/locomotion.md
- ../tasks/manipulation.md
sources:
- ../../sources/repos/simtoolreal.md
- ../../sources/repos/humanoid-motion-intelligence.md
- ../../sources/papers/sun_awesome_r2s2r_2602_16863_simtoolreal-an-object-centric-policy-for.md
- ../../sources/papers/sun_awesome_r2s2r_catalog.md
- ../../sources/repos/awesome-real2sim2real.md
project_id: simtoolreal
arxiv: '2602.16863'
venue: arXiv 2026
---

# SimToolReal

[SimToolReal](https://github.com/tylerlum/simtoolreal) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 的「LocoManip」分组，是本库为该入口建立的独立详情节点。

## 一句话定义

程序化工具随机化几何和动力学，KUKA iiwa14与22自由度SHARPA手跟踪统一6D目标轨迹。140维状态输入和SAPG训练可用于研究跨工具Sim2Real，真实闭环仍依赖外部视觉与机械臂控制仓库。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LocoManip | Loco-Manipulation | 移动与操作同一闭环 |
| RL | Reinforcement Learning | 接触丰富任务的策略学习 |
| WBC | Whole-Body Control | 全身多任务控制 |
| Sim2Real | Simulation to Real | 接触任务迁移到真机 |

| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- **主表工程定位清晰**：该条目被放在「LocoManip」下，说明它服务的是这条人形运动智能问题链上的具体环节，而不是泛泛的链接收藏。
- **可对照开源边界**：主表已概括其可复现范围（训练/推理/部署或仅方法页）；选型时应先读本页「开源状态」，再回官方 README / 项目页核对许可证与平台支持。
- **便于知识库交叉引用**：独立节点让路线图、对比页与 ingest 日志可以稳定链接，避免只在策展列表里「点名」却无法下钻。

## 核心原理

### 在技术路线中的位置

| 字段 | 内容 |
|------|------|
| 主表分组 | LocoManip |
| 官方入口 | https://github.com/tylerlum/simtoolreal |
| 开源状态（据主表） | 已开源（以官方仓库 README 为准） |

主表给出的技术定位可压缩为：

> 程序化工具随机化几何和动力学，KUKA iiwa14与22自由度SHARPA手跟踪统一6D目标轨迹。140维状态输入和SAPG训练可用于研究跨工具Sim2Real，真实闭环仍依赖外部视觉与机械臂控制仓库。

阅读时建议抓住三点：**(1) 输入是什么数据或观测；(2) 输出是参考轨迹、策略、数据还是中间件能力；(3) 公开材料能否支撑训练/部署复现。**

### 流程直觉（对照主表叙事）

```mermaid
flowchart LR
  A["上游数据 / 观测 / 配置"] --> B["SimToolReal"]
  B --> C["下游策略 / 部署 / 评测"]
```

具体模块边界以官方文档为准；本页不替代 README。

## 工程实践

1. **先核入口类型**：若是 GitHub/Gitee 仓库，从 README 的安装、训练与部署章节入手；若是项目页/论文，先确认是否已挂代码或权重。
2. **对齐本体与接口**：人形项目需核对关节顺序、控制频率、观测契约与仿真后端（Isaac / MuJoCo 等）是否与本机栈一致。
3. **按主表定位做消融**：主表强调的可分拆实验切口（例如只换重定向约束、只换部署层）应优先验证，避免一上来全链路重训。
4. **记录开源边界**：若仅有权重、Sim2Sim 或说明文档，不要假设训练管线可复现。

| 检查项 | 建议 |
|--------|------|
| 许可与星标时效 | 以官方仓库页面为准 |
| 支持机器人 / 仿真 | 读 assets 与 task 配置 |
| 真机入口 | 查找 SDK、ROS、ONNX/JIT 导出说明 |

## 局限与风险

- **主表是策展摘要**：细节、指标与许可以一手来源为准；本页只做知识库节点与导航。
- **开源状态可能变化**：标为待发布的项目后续可能放码；已开源仓库也可能拆分或迁移路径。
- **不要与同名论文页混淆**：若本库另有 `paper-*` 深读页，以论文页承载方法细节，本实体页侧重工程入口与选型。

## 关联页面

- [loco-manipulation](../tasks/loco-manipulation.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [开源主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)

- 列表实体：[Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md)
- 技术地图：[Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md)
- 方法/任务：[reinforcement-learning.md](../methods/reinforcement-learning.md)、[locomotion.md](../tasks/locomotion.md)

- [humanoid-motion-intelligence](../entities/humanoid-motion-intelligence.md)
- [crisp-real2sim](../methods/crisp-real2sim.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [SimToolReal 来源归档](../../sources/repos/simtoolreal.md)
- [Humanoid Motion Intelligence 仓库归档](../../sources/repos/humanoid-motion-intelligence.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

- [`sources/papers/sun_awesome_r2s2r_2602_16863_simtoolreal-an-object-centric-policy-for.md`](../../sources/papers/sun_awesome_r2s2r_2602_16863_simtoolreal-an-object-centric-policy-for.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2602.16863>

## 推荐继续阅读

- [官方入口](https://github.com/tylerlum/simtoolreal)
- [Humanoid Motion Intelligence 知识库实体页](./humanoid-motion-intelligence.md)

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2602.16863)
