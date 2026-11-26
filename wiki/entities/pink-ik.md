---
type: entity
tags: [sim2real, tooling, deployment, hmi-opensource-table, repo, inria]
status: draft
updated: 2026-11-18
summary: "Pink：在Pinocchio运动学之上提供轻量级任务空间逆运动学，用加权任务和关节限制求广义速度；适合交互式姿态生成、重定向原型和控制前处理，代码结构便于增加自定义任务。"
related:
  - ../concepts/sim2real.md
  - ../entities/isaac-lab.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/hmi-opensource-projects-coverage.md
  - ../formalizations/inverse-kinematics.md
  - ../concepts/null-space-control.md
sources:
  - ../../sources/repos/pink-ik.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# Pink

[Pink](https://github.com/stephane-caron/pink) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 的「工程与实机部署」分组，是本库为该入口建立的独立详情节点。

## 一句话定义

在Pinocchio运动学之上提供轻量级任务空间逆运动学，用加权任务和关节限制求广义速度；适合交互式姿态生成、重定向原型和控制前处理，代码结构便于增加自定义任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真到真机部署主线 |
| RL | Reinforcement Learning | 训练与评测常用框架 |
| API | Application Programming Interface | 仿真/中间件编程接口 |
| SDK | Software Development Kit | 真机与工具链开发套件 |

## 为什么重要

- **主表工程定位清晰**：该条目被放在「工程与实机部署」下，说明它服务的是这条人形运动智能问题链上的具体环节，而不是泛泛的链接收藏。
- **可对照开源边界**：主表已概括其可复现范围（训练/推理/部署或仅方法页）；选型时应先读本页「开源状态」，再回官方 README / 项目页核对许可证与平台支持。
- **便于知识库交叉引用**：独立节点让路线图、对比页与 ingest 日志可以稳定链接，避免只在策展列表里「点名」却无法下钻。

## 核心原理

### 在技术路线中的位置

| 字段 | 内容 |
|------|------|
| 主表分组 | 工程与实机部署 |
| 官方入口 | https://github.com/stephane-caron/pink |
| 开源状态（据主表） | 已开源（以官方仓库 README 为准） |

主表给出的技术定位可压缩为：

> 在Pinocchio运动学之上提供轻量级任务空间逆运动学，用加权任务和关节限制求广义速度；适合交互式姿态生成、重定向原型和控制前处理，代码结构便于增加自定义任务。

阅读时建议抓住三点：**(1) 输入是什么数据或观测；(2) 输出是参考轨迹、策略、数据还是中间件能力；(3) 公开材料能否支撑训练/部署复现。**

### 流程直觉（对照主表叙事）

```mermaid
flowchart LR
  A["上游数据 / 观测 / 配置"] --> B["Pink"]
  B --> C["下游策略 / 部署 / 评测"]
```

具体模块边界以官方文档为准；本页不替代 README。

## 源码运行时序图

以下时序对照 `stephane-caron/pink`（`main` 分支）中 [examples/arm_ur5.py](https://github.com/stephane-caron/pink/blob/main/examples/arm_ur5.py)、[pink/configuration.py](https://github.com/stephane-caron/pink/blob/main/pink/configuration.py)、[pink/solve_ik.py](https://github.com/stephane-caron/pink/blob/main/pink/solve_ik.py) 与 [pink/tasks/frame_task.py](https://github.com/stephane-caron/pink/blob/main/pink/tasks/frame_task.py)，展示单步微分逆运动学闭环：

```mermaid
sequenceDiagram
  autonumber
  participant CtrlLoop as "控制循环 (如 examples/arm_ur5.py)"
  participant Config as "pink.Configuration (Pinocchio 封装)"
  participant Tasks as "Task / Limit / Barrier"
  participant SolveIK as "pink.solve_ik / build_ik"
  participant QP as "qpsolvers.solve_problem"

  CtrlLoop->>Config: "Configuration(robot.model, robot.data, q_ref)"
  Config->>Config: "pin.computeJointJacobians + updateFramePlacements (+ 碰撞距离)"
  CtrlLoop->>Tasks: "task.set_target_from_configuration(configuration)"

  loop "每个控制周期 dt"
    CtrlLoop->>Tasks: "更新目标位姿 (如 transform_target_to_world)"
    CtrlLoop->>SolveIK: "solve_ik(configuration, tasks, dt, solver=solver)"
    SolveIK->>Config: "check_limits(safety_break)"
    SolveIK->>Tasks: "__compute_qp_objective: 汇总 task.compute_qp_objective(configuration)"
    SolveIK->>Tasks: "__compute_qp_inequalities: 汇总 Limit/Barrier (G*dq <= h)"
    SolveIK->>Tasks: "__compute_qp_equalities: 汇总硬约束 (A*dq = b)"
    SolveIK->>QP: "qpsolvers.solve_problem(Problem(P, q, G, h, A, b), solver)"
    QP-->>SolveIK: "result.x (构型增量 Delta_q)"
    SolveIK-->>CtrlLoop: "返回切空间速度 v = Delta_q / dt"
    CtrlLoop->>Config: "integrate_inplace(v, dt)"
    Config->>Config: "pin.integrate(model, q, v * dt) 并 update(q) 刷新 FK/雅可比"
  end
```

> **源码映射说明**：`solve_ik` 与 QP 装配函数（`build_ik`、`__compute_qp_objective`、`__compute_qp_inequalities`、`__compute_qp_equalities`）位于 `pink/solve_ik.py`；Pinocchio 正向运动学/雅可比与流形积分封装位于 `pink/configuration.py`；典型实时跟踪示例见 `examples/arm_ur5.py`。

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

- [sim2real](../concepts/sim2real.md)
- [isaac-lab](../entities/isaac-lab.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [逆运动学](../formalizations/inverse-kinematics.md) — 任务空间 IK 与 Pinocchio 几何雅可比
- [零空间控制](../concepts/null-space-control.md) — 冗余次级任务；Pink 用加权任务而非手写 $N$
- [开源主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)
- [GMR: 通用动作重定向](../methods/motion-retargeting-gmr.md) — 两者都涉及任务空间误差与差分 IK，可对照加权任务、关节限位及人体到机器人映射。
- [Manipulation](../tasks/manipulation.md) — 机械臂姿态生成可作为操作前处理，但 IK 可解不保证避碰或接触稳定。

## 参考来源

- [Pink 来源归档](../../sources/repos/pink-ik.md)
- [Humanoid Motion Intelligence 仓库归档](../../sources/repos/humanoid-motion-intelligence.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

## 推荐继续阅读

- [官方入口](https://github.com/stephane-caron/pink)
- [Humanoid Motion Intelligence 知识库实体页](./humanoid-motion-intelligence.md)
