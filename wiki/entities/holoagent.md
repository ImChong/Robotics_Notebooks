---
type: entity
tags: [vla, world-model, imitation-learning, hmi-opensource-table, repo, horizon-robotics]
status: draft
updated: 2026-07-30
summary: "HoloAgent：AgentOS把语言任务展开为受监控的技能图，三维空间记忆负责保存环境变化并在失败后触发重规划。当前开放的导航组件已经露出记忆更新、技能调用和异常恢复之间的接口，新能力仍有一部分停留在发布路线图。"
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/hmi-opensource-projects-coverage.md
  - ./robo-orchard-lab.md
sources:
  - ../../sources/repos/holoagent.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# HoloAgent

[HoloAgent](https://github.com/HorizonRobotics/HoloAgent) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 的「世界模型、VLA与Agent」分组，是本库为该入口建立的独立详情节点。

## 一句话定义

AgentOS把语言任务展开为受监控的技能图，三维空间记忆负责保存环境变化并在失败后触发重规划。当前开放的导航组件已经露出记忆更新、技能调用和异常恢复之间的接口，新能力仍有一部分停留在发布路线图。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言条件动作策略 |
| IL | Imitation Learning | 从示范学习控制 |
| WM | World Model | 预测未来观测/状态的模型 |
| BC | Behavior Cloning | 监督式模仿基线 |

## 为什么重要

- **主表工程定位清晰**：该条目被放在「世界模型、VLA与Agent」下，说明它服务的是这条人形运动智能问题链上的具体环节，而不是泛泛的链接收藏。
- **可对照开源边界**：主表已概括其可复现范围（训练/推理/部署或仅方法页）；选型时应先读本页「开源状态」，再回官方 README / 项目页核对许可证与平台支持。
- **便于知识库交叉引用**：独立节点让路线图、对比页与 ingest 日志可以稳定链接，避免只在策展列表里「点名」却无法下钻。

## 核心原理

### 在技术路线中的位置

| 字段 | 内容 |
|------|------|
| 主表分组 | 世界模型、VLA与Agent |
| 官方入口 | https://github.com/HorizonRobotics/HoloAgent |
| 开源状态（据主表） | 已开源（以官方仓库 README 为准） |

主表给出的技术定位可压缩为：

> AgentOS把语言任务展开为受监控的技能图，三维空间记忆负责保存环境变化并在失败后触发重规划。当前开放的导航组件已经露出记忆更新、技能调用和异常恢复之间的接口，新能力仍有一部分停留在发布路线图。

阅读时建议抓住三点：**(1) 输入是什么数据或观测；(2) 输出是参考轨迹、策略、数据还是中间件能力；(3) 公开材料能否支撑训练/部署复现。**

### 流程直觉（对照主表叙事）

```mermaid
flowchart LR
  A["上游数据 / 观测 / 配置"] --> B["HoloAgent"]
  B --> C["下游策略 / 部署 / 评测"]
```

具体模块边界以官方文档为准；本页不替代 README。

## 源码运行时序图

以下时序图基于官方仓库 `HorizonRobotics/HoloAgent`（`main` 分支）中已开放的 AgentOS 编排、后台守护与 HTTP↔ROS 2 桥接代码核读整理：
- 核对源码路径：`agentic_robot/agentOS/sandbox_test/long_horizon_text_runner.py`、`agentic_robot/services/src/robot_bridge/robot_bridge/robot_bridge_node.py`、`agentic_robot/agentOS/holoagent_skills/skills/sem-nav-skill/scripts/semantic_nav.py`、`agentic_robot/agentOS/run_dameon/run_g1_background_daemon.py`、`scripts/intergation/run_holoagent_pipeline.sh`

```mermaid
sequenceDiagram
  autonumber
  participant Launcher as "run_holoagent_pipeline.sh / run_g1_background_daemon.py"
  participant Runner as "LongHorizonTextRunner (long_horizon_text_runner.py)"
  participant LLM as "AzureOpenAI / OpenAI Client"
  participant Ctrl as "Control Center (:8080)"
  participant Bridge as "RobotBridgeNode + FastAPI (robot_bridge_node.py)"
  participant ROS as "ROS 2 Nodes (nav / arm / relative_goal)"

  Launcher->>ROS: 启动定位、导航、传感器、运控与 relative_goal_node
  Runner->>Runner: run(instruction)
  Runner->>LLM: analyze_instruction() 请求 responses.create()
  LLM-->>Runner: DAG JSON -> 写入 dag_plan.yaml
  Runner->>Runner: virtual_validate(dag) -> 写入 virtual_validation.yaml
  Runner->>Runner: execute(dag) via ThreadPoolExecutor
  loop DAG 就绪节点 (navigation / arm)
    alt skill == navigation
      Runner->>Ctrl: POST /reset_reached
      Runner->>Bridge: HTTP POST 导航指令
      Bridge->>Bridge: _build_msg() 构造 ROS 消息
      Bridge->>ROS: publish_to_topic() 发布导航话题
      loop 轮询导航完成
        Runner->>Ctrl: GET /robot_reached/{robot_id}
        ROS-->>Bridge: 订阅 reverse 话题到达事件
        Bridge-->>Ctrl: requests.post(http_post) 上报状态
      end
    else skill == arm
      Runner->>Bridge: HTTP POST 触发手臂技能
      Bridge->>ROS: publish_to_topic() 发布运控话题
      Runner->>Runner: 等待 _ARM_SKILL_WAIT_SEC 并写 monitor.yaml
    end
  end
```

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

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [开源主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)

## 参考来源

- [HoloAgent 来源归档](../../sources/repos/holoagent.md)
- [Humanoid Motion Intelligence 仓库归档](../../sources/repos/humanoid-motion-intelligence.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

## 推荐继续阅读

- [官方入口](https://github.com/HorizonRobotics/HoloAgent)
- [Humanoid Motion Intelligence 知识库实体页](./humanoid-motion-intelligence.md)
