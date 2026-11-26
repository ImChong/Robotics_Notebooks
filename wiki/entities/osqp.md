---
type: entity
tags: [sim2real, tooling, deployment, hmi-opensource-table, repo, linux-foundation]
status: complete
updated: 2026-11-21
summary: "OSQP：用算子分裂求解稀疏凸二次规划，并支持热启动、代码生成与多语言绑定；在人形控制中常承载接触力分配、约束逆动力学和MPC子问题，稳定的状态复用有利于实时循环。"
related:
  - ../concepts/sim2real.md
  - ../entities/isaac-lab.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/hmi-opensource-projects-coverage.md
sources:
  - ../../sources/repos/osqp.md
  - ../../sources/repos/humanoid-motion-intelligence.md
original:
  - ../../sources/repos/osqp.md
---

# OSQP

[OSQP](https://github.com/osqp/osqp) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 的「工程与实机部署」分组，是本库为该入口建立的独立详情节点。

## 一句话定义

用算子分裂求解稀疏凸二次规划，并支持热启动、代码生成与多语言绑定；在人形控制中常承载接触力分配、约束逆动力学和MPC子问题，稳定的状态复用有利于实时循环。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OSQP | OSQP | OSQP 相关缩写，详见正文 |
| Sim2Real | Simulation to Real | 仿真到真机部署主线 |
| RL | Reinforcement Learning | 训练与评测常用框架 |
| API | Application Programming Interface | 仿真/中间件编程接口 |

## 为什么重要

- **主表工程定位清晰**：该条目被放在「工程与实机部署」下，说明它服务的是这条人形运动智能问题链上的具体环节，而不是泛泛的链接收藏。
- **可对照开源边界**：主表已概括其可复现范围（训练/推理/部署或仅方法页）；选型时应先读本页「开源状态」，再回官方 README / 项目页核对许可证与平台支持。
- **便于知识库交叉引用**：独立节点让路线图、对比页与 ingest 日志可以稳定链接，避免只在策展列表里「点名」却无法下钻。

## 核心原理

### 在技术路线中的位置

| 字段 | 内容 |
|------|------|
| 主表分组 | 工程与实机部署 |
| 官方入口 | https://github.com/osqp/osqp |
| 开源状态（据主表） | 已开源（以官方仓库 README 为准） |

主表给出的技术定位可压缩为：

> 用算子分裂求解稀疏凸二次规划，并支持热启动、代码生成与多语言绑定；在人形控制中常承载接触力分配、约束逆动力学和MPC子问题，稳定的状态复用有利于实时循环。

阅读时建议抓住三点：**(1) 输入是什么数据或观测；(2) 输出是参考轨迹、策略、数据还是中间件能力；(3) 公开材料能否支撑训练/部署复现。**

### 流程直觉（对照主表叙事）

```mermaid
flowchart LR
  A["上游数据 / 观测 / 配置"] --> B["OSQP"]
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

## 源码运行时序图

入口：C API 典型调用链（见 `docs/examples/setup-and-solve.rst`）：`osqp_setup` → `osqp_solve` →（可选参数更新复用分解）`osqp_update_data_vec` / `osqp_warm_start` → `osqp_solve` → `osqp_cleanup`（Python/Matlab 等绑定最终转发至同一 C API）。

```mermaid
sequenceDiagram
    autonumber
    participant Caller as 调用方 (C / Python 绑定)
    participant API as src/osqp_api.c
    participant Aux as src/auxil.c ADMM 核心
    participant Lin as linsys_solver (QDLDL / MKL / CUDA)

    Caller->>API: OSQPCscMatrix_new(P, A) 与 OSQPSettings_new()
    Caller->>API: osqp_setup(&solver, P, q, A, l, u, m, n, settings)
    API->>API: validate_data 与 validate_settings
    API->>API: osqp_algebra_init_libs 并复制 P, q, A, l, u 至 work->data
    opt settings->scaling 开启
        API->>API: scale_data(solver)
    end
    API->>Lin: osqp_algebra_init_linsys_solver 构造并分解 KKT 系统
    API->>API: osqp_cold_start(solver)
    API-->>Caller: exitflag = 0

    Caller->>API: osqp_solve(solver)
    opt !settings->warm_starting
        API->>API: osqp_cold_start(solver)
    end
    loop iter = 1 .. settings->max_iter
        API->>Aux: update_xz_tilde(solver, iter)
        Aux->>Lin: work->linsys_solver->solve 求解 KKT 线性系统
        API->>Aux: update_x(solver) (松弛参数 alpha)
        API->>Aux: update_z(solver) (投影至 [l, u])
        API->>Aux: update_y(solver) (更新对偶变量)
        opt 检查终止或自适应 rho
            API->>Aux: update_info 与 check_termination (达标则 break)
            API->>Aux: adapt_rho (必要时重分解 KKT)
        end
    end
    opt settings->polishing && OSQP_SOLVED
        API->>API: polish(solver)
    end
    API->>Aux: store_solution(solver, solver->solution)
    API-->>Caller: solver->solution (x, y)

    opt 实时控制复用循环 (向量更新与热启动)
        Caller->>API: osqp_update_data_vec(solver, q_new, l_new, u_new)
        Caller->>API: osqp_warm_start(solver, x, y)
        Caller->>API: osqp_solve(solver)
    end
    Caller->>API: osqp_cleanup(solver)
```

图依据：`osqp/osqp`（`master` 分支）实际核读 `README.md`、`docs/examples/setup-and-solve.rst`、`include/public/osqp_api_functions.h`、`src/osqp_api.c` 与 `src/auxil.c`；未进行本地编译运行。

## 关联页面

- [sim2real](../concepts/sim2real.md)
- [isaac-lab](../entities/isaac-lab.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [开源主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)

## 参考来源

- [OSQP 来源归档](../../sources/repos/osqp.md)
- [Humanoid Motion Intelligence 仓库归档](../../sources/repos/humanoid-motion-intelligence.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

## 推荐继续阅读

- [官方入口](https://github.com/osqp/osqp)
- [Humanoid Motion Intelligence 知识库实体页](./humanoid-motion-intelligence.md)
