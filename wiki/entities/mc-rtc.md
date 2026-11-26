---
type: entity
tags: [sim2real, tooling, deployment, hmi-opensource-table, repo, inria]
status: draft
updated: 2026-07-30
summary: "mc_rtc：有限状态机负责行为切换，任务栈和接触管理在实时循环内共同求解姿态、行走与操作目标。日志、GUI和机器人模块把学习策略接入WBC时的命令、约束和失败位置完整暴露出来。"
related:
  - ../concepts/sim2real.md
  - ../entities/isaac-lab.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/hmi-opensource-projects-coverage.md
sources:
  - ../../sources/repos/mc-rtc.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# mc_rtc

[mc_rtc](https://github.com/jrl-umi3218/mc_rtc) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 的「工程与实机部署」分组，是本库为该入口建立的独立详情节点。

## 一句话定义

有限状态机负责行为切换，任务栈和接触管理在实时循环内共同求解姿态、行走与操作目标。日志、GUI和机器人模块把学习策略接入WBC时的命令、约束和失败位置完整暴露出来。

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
| 官方入口 | https://github.com/jrl-umi3218/mc_rtc |
| 开源状态（据主表） | 已开源（以官方仓库 README 为准） |

主表给出的技术定位可压缩为：

> 有限状态机负责行为切换，任务栈和接触管理在实时循环内共同求解姿态、行走与操作目标。日志、GUI和机器人模块把学习策略接入WBC时的命令、约束和失败位置完整暴露出来。

阅读时建议抓住三点：**(1) 输入是什么数据或观测；(2) 输出是参考轨迹、策略、数据还是中间件能力；(3) 公开材料能否支撑训练/部署复现。**

### 流程直觉（对照主表叙事）

```mermaid
flowchart LR
  A["上游数据 / 观测 / 配置"] --> B["mc_rtc"]
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

入口：独立周期运行工具 `mc_rtc_ticker`（`utils/mc_rtc_ticker.cpp`），或由外部仿真/真机驱动周期调用 `MCGlobalController::setEncoderValues` 与 `MCGlobalController::run`。

```mermaid
sequenceDiagram
    autonumber
    participant Main as utils/mc_rtc_ticker.cpp main
    participant Tk as src/mc_control/Ticker.cpp Ticker
    participant GC as mc_global_controller.cpp MCGlobalController
    participant FSM as fsm/Controller.cpp & Executor.cpp
    participant MC as MCController.cpp & qpsolver
    participant Srv as ControllerServer & Logger

    Main->>Tk: 构造 Ticker(config) 并调用 ticker.run()
    Tk->>GC: 构造 MCGlobalController (加载全局插件与控制器插件)
    Tk->>Tk: simulate_sensors() 注入初始编码器/基座状态
    Tk->>GC: gc_.init() 触发 initController() 与 resetObserverPipelines()
    loop Ticker::run() 步进循环
        Tk->>Tk: step() 调用 simulate_sensors() 更新传感器输入
        Tk->>GC: gc_.run()
        GC->>GC: 检查 next_controller_ 热切换并调用 plugins_before_->before()
        GC->>MC: controller_->runObserverPipelines()
        GC->>FSM: controller_->run()
        alt 控制器为 fsm::Controller
            FSM->>FSM: executor_.run() 执行 state_->run() 与状态转移检查
            FSM->>MC: MCController::run(fType)
        end
        MC->>MC: updateContacts() 与 qpsolver->run(fType) 求解 QP
        GC->>GC: converters_ 转换输出机器人并执行 grippers->run()
        GC->>Srv: server_->handle_requests/publish 与 logger().log()
        GC->>GC: plugins_after_->after()
    end
```

图依据：`jrl-umi3218/mc_rtc`（`master` 分支）实际核读 `README.md`、`utils/mc_rtc_ticker.cpp`、`src/mc_control/Ticker.cpp`、`src/mc_control/mc_global_controller.cpp`、`src/mc_control/fsm/Controller.cpp`、`src/mc_control/fsm/Executor.cpp` 与 `src/mc_control/MCController.cpp`；未进行本地编译运行。

## 关联页面

- [sim2real](../concepts/sim2real.md)
- [isaac-lab](../entities/isaac-lab.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [开源主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)

## 参考来源

- [mc_rtc 来源归档](../../sources/repos/mc-rtc.md)
- [Humanoid Motion Intelligence 仓库归档](../../sources/repos/humanoid-motion-intelligence.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

## 推荐继续阅读

- [官方入口](https://github.com/jrl-umi3218/mc_rtc)
- [Humanoid Motion Intelligence 知识库实体页](./humanoid-motion-intelligence.md)
