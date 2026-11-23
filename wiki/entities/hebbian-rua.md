---
type: entity
project_id: hebbian-rua
tags: [project, robot-agent, manipulation, teleoperation, llm, robot-control]
status: complete
updated: 2026-11-23
project: https://hebbianrobotics.com/
code: https://github.com/Hebbian-Robotics/rua
original:
  - ../../sources/repos/hebbian-rua.md
sources:
  - ../../sources/repos/hebbian-rua.md
  - ../../sources/sites/hebbian-rua.md
related:
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/vla.md
  - ../concepts/llm-robotics-control-interfaces.md
  - ../../roadmap/depth-teleoperation.md
summary: "RUA 将相机和机器人状态以工具接口暴露给 LLM，依靠 SDK、Compiler、Runtime 和受控 motor controller 执行单/双臂 SO-101 任务。"
---

# Hebbian RUA：基于工具编排的机器人使用智能体 / Hebbian Robot-Use Agents

## 一句话定义

Hebbian Robot-Use Agents（RUA）提供可观测、可调用和可执行的机器人工具链：应用自己负责大模型决策循环，RUA 负责机器人状态、动作编译、运行时检查和电机指令交付。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| RUA | Robot-Use Agents | 机器人使用智能体框架 |
| LLM | Large Language Model | 通过工具调用规划下一步动作 |
| SDK | Software Development Kit | Python、TypeScript、Rust 的客户端接口 |
| API | Application Programming Interface | Station 提供机器人工具与命令访问 |

## 核心信息

| 项 | 资料 |
|---|---|
| 官方入口 | [项目或论文](https://hebbianrobotics.com/) |
| 官方源码 | [GitHub](https://github.com/Hebbian-Robotics/rua) |
| 开放程度 | 官方代码/资料可访问，详见归档与许可边界 |
| 本次检查 | 2026-11-23，一手材料静态核对，未独立复现 |

## 为什么重要

文本指令不能直接等同为安全的关节指令。将 LLM 的工具调用和实际机器人执行用编译、运行时重校验、权限/接管机制隔开，能提高可观察性和控制边界的清晰度。

## 核心机制

1. SDK 获取相机帧、关节状态和动作结果，组成下轮模型观测。
2. 应用将用户目标和可用机器人工具交给 LLM；模型返回工具调用而非直接接管驱动器。
3. Station 中 Robot Use Compiler 校验动作序列并生成编译结果；Runtime 根据实时状态准备步骤，再经 command admission 和 motor controller 发给机器人。
4. 执行状态与新的图像反馈 SDK，构成「观察—决策—执行—再观察」闭环。

## 流程总览

```mermaid
flowchart TD
  Goal["用户自然语言目标"] --> Loop["Agent / LLM 决策循环"]
  Camera["相机帧与关节状态"] --> SDK["RUA SDK 观测和工具接口"]
  SDK --> Loop
  Loop --> Tools["LLM 选择机器人工具"]
  Tools --> API["Station API"]
  API --> Compiler["Robot Use Compiler"]
  Compiler --> Runtime["Robot Use Runtime"]
  Runtime --> Admission["Command Admission"]
  Admission --> Motor["Motor Controller"]
  Motor --> Arm["SO-101 单臂或双臂 / MuJoCo"]
  Arm --> Camera
```

## 源码运行时序图

依据仓库 README、官方可运行入口及相应模块文件编写；仅核对静态源码，未声称已执行训练或真机运行。

```mermaid
sequenceDiagram
  autonumber
  actor User as 用户
  participant App as examples/python/agent_quickstart.py
  participant SDK as OpenAIRobotTools
  participant LLM as LLM Responses API
  participant Station as Station API
  participant Compiler as Robot Use Compiler
  participant Runtime as Robot Use Runtime
  participant Motor as Motor Controller
  participant Arm as SO-101
  User->>App: 提供任务目标
  loop 最多 max_turns 轮
    App->>SDK: observe(wait_for_recovery=True)
    SDK-->>App: 相机观测、机器人状态及 revisions
    App->>LLM: messages + robot_tools.tools
    LLM-->>App: 工具调用
    App->>SDK: execute(response, expected_revisions=...)
    SDK->>Station: 请求执行动作序列
    Station->>Compiler: 检查与翻译命令
    Compiler->>Runtime: 编译步骤
    Runtime->>Motor: 通过 admission 交付动作
    Motor->>Arm: 执行电机动作
    Arm-->>Station: 状态与结果
    Station-->>SDK: 工具执行结果
    SDK-->>App: 本轮结果供下轮观察
  end
```

## 项目资料与能力边界

| 维度 | 官方公开状态 |
|---|---|
| 受支持机器人 | 单臂和双臂 SO-101 |
| 仿真 | 提供 MuJoCo 测试路径 |
| 客户端 | Python / Rust / TypeScript SDK；模型由用户选择 |
| 授权 | MIT；第三方依赖另遵其各自许可 |

## 结论

**Hebbian Robot-Use Agents（RUA）提供可观测、可调用和可执行的机器人工具链：应用自己负责大模型决策循环，RUA 负责机器人状态、动作编译、运行时检查和电机指令交付；使用时必须区分作者报告和可独立复现的验证结果。**

1. LLM 不应直接发送电机原始指令，需经过运行时许可和安全检查。
2. 将模型推理循环与机器人 Station 的实时控制职责分离。
3. 先用 MuJoCo、再用受控网络与实体急停监督下的小幅真机动作验证。

## 工程实践

官方 `examples/python/agent_quickstart.py` 中 `OpenAIRobotTools.observe()`、`client.responses.create()`、`robot_tools.execute()` 构成可核对循环；`docs/architecture.md` 明确 Station API → Compiler → Runtime → command admission → motor controller。官方 CLI 有 `uv run rua serve --help`。建议先按 `docs/simulation.md` 在 MuJoCo 测试，再监督真机。


## 与其他工具对比

| 路线 | 输出 | 边界 |
|---|---|---|
| [VLA](../methods/vla.md) | 直接输出连续动作/动作块 | RUA 主要执行模型选择的预定义工具 |
| [Mind2Act World](./paper-mind2act-world.md) | 对执行与再规划做任务评测 | RUA 提供工具化运行时而非通用 benchmark |

## 局限与风险

真实机械臂可被工具调用移动，必须使用受控网络、明确的租约和硬件断电手段；适配器输出不是形式化安全保证。当前文档明确支持 SO-101，不应推断对其他硬件自动兼容。

## 关联页面

- [相关知识与路线](../tasks/manipulation.md)
- [相关知识与路线](../tasks/teleoperation.md)
- [相关知识与路线](../methods/vla.md)
- [相关知识与路线](../concepts/llm-robotics-control-interfaces.md)
- [相关知识与路线](../../roadmap/depth-teleoperation.md)

## 参考来源

- [原始资料归档](../../sources/repos/hebbian-rua.md)
- [原始资料归档](../../sources/sites/hebbian-rua.md)

## 推荐继续阅读

- [RUA 官方代码](https://github.com/Hebbian-Robotics/rua)
- [运行架构](https://github.com/Hebbian-Robotics/rua/blob/main/docs/architecture.md)
