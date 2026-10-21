# Scalable Multi-Human Cooperative Manipulation via Commander-Executor Coordination

> 来源归档（ingest）

- **类型：** 论文（arXiv 预印本）
- **机构：** 天津大学（TJU）
- **作者：** Buzhen Huang、Xingchen Wu、Yuanwang Yang、Chenyi Xie、Kun Li
- **论文：** <https://arxiv.org/abs/2610.14911>
- **作者发表列表：** <https://www.buzhenhuang.com/publications/>
- **提交日期：** 2026-10-10
- **入库日期：** 2026-10-19
- **项目/代码：** 论文称代码将发布到 <https://github.com/TJU-3DV/Co-Manipulation>；截至入库日仓库未公开可访问，作者发表列表中的 project / paper / code 链接也为空。
- **一句话说明：** 归档一个由高层语义与几何规划器分配角色、低层共享物理策略执行的多角色人形角色协作操作框架。

## 核心论文摘录

### Scalable Multi-Human Cooperative Manipulation via Commander-Executor Coordination（Buzhen Huang 等，2026）
- **链接：** <https://arxiv.org/abs/2610.14911>
- **核心贡献：** Commander 使用场景图与任务指令做语义分解，再以 A* / Hybrid A*、退火采样等几何规划生成物体轨迹、站位、接触点与抓放信号；Executor 先在增强的单角色操作任务上训练，再微调到多角色协作。所有角色共享同一个 PPO 物理策略，并通过角色专属局部观测和共同物体轨迹协调。论文报告其在 Core4D 轨迹上的双角色搬运成功率为 80.26%，在 Bezier 轨迹上为 86.50%，并展示接力交接与最多五角色的协作。
- **范围与限制：** 论文生成的是物理仿真中的 humanoid character 交互动作，不能据此视作真实机器人部署验证；作者指出简化球形手模型限制细长物体灵巧操作，复杂地形与丰富环境接触尚未纳入低层执行。
- **对 wiki 的映射：**
  - [Commander–Executor 协作操作实体页](../../wiki/entities/paper-commander-executor-coordination.md)
  - [Manipulation 任务页](../../wiki/tasks/manipulation.md)

## 当前提炼状态

- [x] 论文信息、方法和评测摘录
- [x] wiki 页面映射确认
- [x] 代码开放状态核查：论文承诺发布，仓库截至 2026-10-19 未公开可访问
