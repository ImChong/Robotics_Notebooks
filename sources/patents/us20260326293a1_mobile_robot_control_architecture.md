# Control Architecture for a Mobile Robot — Boston Dynamics（US2026/0326293 A1）

> 专利申请来源归档（ingest）

- **类型：** patent / control architecture / humanoid / mobile manipulation
- **申请人：** Boston Dynamics, Inc.
- **发明人：** Frans Anton Koolen、Michael Lutter、Scott Kuindersma、Alberto Rodriguez
- **美国申请号：** 19/419,924
- **公布号：** US 2026/0326293 A1（US20260326293A1）
- **申请日：** 2025-12-15
- **公布日：** 2026-10-22
- **同族 PCT：** PCT/US2025/059651；WO2026/136238 A1，2026-06-25 公布
- **优先权：** US 63/736,965，2024-12-20
- **一句话说明：** 提出将机器人全身动态平衡控制与局部末端执行器控制分开运行，并通过动力学协调模块和接口约定协同，使机器人可以在保持全身控制（例如行走、平衡）的同时执行手臂操作。

## 专利披露要点

1. **双控制器分工：** 第一控制器负责移动机器人的全身动态平衡；第二控制器负责机器人某一局部（尤其末端执行器）的运动。
2. **动力学协调模块：** 在两类控制器间协调操作。公开文本给出的接口例子包括阻抗信息；控制器可据此交换约束或参考信息。
3. **局部策略与全身控制解耦：** 末端执行器可由独立训练的学习策略控制，再向全身控制器提供末端轨迹参考。专利列举强化学习、行为克隆等策略学习方式。
4. **全身控制器实现：** 可采用模型驱动控制（例如 MPC），也可采用学习策略；描述的输出包括关节位置/力矩、质心运动及接触力/力矩等。
5. **并行操作与运动：** 所述方法允许局部控制器操控末端执行器与环境交互，同时由全身控制器处理机器人整体动态。

## 阅读边界

这是专利申请中的权利要求与实施方式描述。上述架构是申请人披露的方案，不足以单独证明其已经部署在 Atlas、Spot 或其他具体产品中，也不代表专利已获授权。

## 一手来源

- [USPTO 公布文本 PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/20260326293)
- [USPTO Patent Center：申请 19/419,924](https://patentcenter.uspto.gov/applications/19419924)
- [WIPO PATENTSCOPE：国际申请查询入口](https://patentscope.wipo.int/search/en/structuredSearch.jsf?query=PCT%2FUS2025%2F059651)
- [同族国际公布文本 WO2026/136238 A1（可检索文本）](https://patents.google.com/patent/WO2026136238A1/en)

## 资料映射

- sources/patents/boston_dynamics_legged_robot_patents.md — Boston Dynamics 足式机器人控制与硬件专利摘录
- wiki/entities/patent-boston-dynamics-legged-control-stack.md — Boston Dynamics 足式机器人控制栈专利映射（后续可补入本申请）
