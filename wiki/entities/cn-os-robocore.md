---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/robocore.md
summary: "玄雅科技 开源项目 RoboCore（工程与工具）：Synria 统一高吞吐机器人开发库…"
institutions:
  - xuanya
---

# RoboCore

## 一句话定义

**RoboCore** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **工程与工具** 开源项目：Synria 统一高吞吐机器人开发库

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 玄雅科技 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/Synria-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `RoboCore` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：README Quick Start 脚本（先 `rc.set_backend("cpp")`，再调用 `forward_kinematics` 与 `inverse_kinematics`，均经 `robocore/kinematics/__init__.py` 导出）。

```mermaid
sequenceDiagram
    participant U as 用户脚本
    participant BM as utils/backend.py
    participant RM as modeling/robot_model.py RobotModel
    participant UP as urdf_parser URDFParser
    participant FK as kinematics/fk.py forward_kinematics
    participant FKC as fk_utils/fk_solver_cpp.py FKSolverCpp
    participant FKX as fk_utils/_fk_chain_core C++
    participant IK as kinematics/ik.py inverse_kinematics
    participant IKG as ik_utils/initial_guess.py
    participant IKC as ik_utils/ik_solver_cpp.py IKSolverCpp
    participant IKX as ik_utils/_ik_chain_core C++

    U->>BM: rc.set_backend("cpp") 设置全局后端
    U->>RM: RobotModel(urdf, base_link, end_link)
    RM->>UP: 未命中缓存时 URDFParser 解析 URDF
    UP-->>RM: parsed_model 关节与链信息
    U->>RM: model.random_q(seed=0)
    RM-->>U: 关节配置 q
    U->>FK: forward_kinematics(model, q, return_end=True, backend="cpp")
    FK->>BM: backend_context("cpp") 取得当前后端
    FK->>FKC: 构造 FKSolverCpp(model)
    FKC->>FKX: ChainFK(关节类型, q 索引, 关节原点, 关节轴, 自由度)
    FK->>FKC: solve(q, return_end_only=True)
    FKC->>FKX: fk_end(q) 批量正运动学
    FKX-->>FKC: 末端 4x4 位姿
    FKC-->>FK: 位姿 T
    FK-->>U: T
    U->>IK: inverse_kinematics(model, T, q, method="dls", backend="cpp")
    IK->>BM: backend_context("cpp")
    IK->>IKG: generate_initial_guesses 生成初值
    IK->>IKC: _solve_cpp 构造 IKSolverCpp(max_iters, pos_tol, ori_tol)
    IKC->>IKX: 构造 ChainIKDls
    IKC->>IKX: solve(target, q0, max_iters, 容差) 单目标
    IKX-->>IKC: 求解结果 q 与收敛状态
    IKC-->>IK: 结果字典
    IK-->>U: 结果字典（含 success 与 q）
```

图依据：RoboCore@b77886c，读过 README.md、pyproject.toml、robocore/__init__.py、robocore/kinematics/__init__.py、robocore/kinematics/fk.py、robocore/kinematics/fk_utils/fk_solver_cpp.py、robocore/kinematics/ik.py、robocore/kinematics/ik_utils/ik_solver_cpp.py（部分）、robocore/modeling/robot_model.py（`_load_model` 与 `random_q`）、robocore/utils/backend.py（`set_backend` 段）；`URDFParser`、`initial_guess.py`、C++ 绑定文件仅确认调用点与绑定名，未读实现。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RoboCore 源码归档](../../sources/repos/robocore.md)（<https://github.com/Synria-Robotics/RoboCore>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：Synria-Robotics/RoboCore@b77886c，读过 README.md、pyproject.toml、robocore/__init__.py、robocore/kinematics/__init__.py、robocore/kinematics/fk.py、robocore/kinematics/fk_utils/fk_solver_cpp.py、robocore/kinematics/ik.py、robocore/kinematics/ik_utils/ik_solver_cpp.py、robocore/modeling/robot_model.py、robocore/utils/backend.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
