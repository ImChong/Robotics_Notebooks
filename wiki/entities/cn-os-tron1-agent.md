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
  - ../../sources/repos/tron1-agent.md
summary: "逐际动力 开源项目 tron1-agent（具身Agent/规划）：具身 Agent：任务规划与技能调度…"
institutions:
  - limx
---

# tron1-agent

## 一句话定义

**tron1-agent** 是 [逐际动力](https://github.com/limxdynamics) 公开的 **具身Agent/规划** 开源项目：具身 Agent：任务规划与技能调度

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 逐际动力 |
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/limxdynamics、https://github.com/FluxVLA |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `tron1-agent` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python main.py --config_path config/config.yaml`（[start.sh](start.sh) 以无参数方式启动，默认读取 config/config.yaml）

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户（语音）
    participant MA as main.py
    participant R as Robot（bailing/robot.py）
    participant ASR as FunASR（语音识别）
    participant TM as TaskManager（plugins/task_manager.py）
    participant LLM as OpenAILLM（bailing/llm.py，Ollama）
    participant TA as TronAction（plugins/functions/tron_action.py）
    participant FL as pocket Flow（third_party/tron）
    participant WS as TronWebsocket（third_party/tron/communication）
    participant TTS as PiperTTS 与播放器
    participant RB as TRON1 机器人 ws://ip:5000（仓库外）

    MA->>R: Robot(config_path)
    R->>TM: call_tools init_client 异步建立连接
    TM->>WS: TronWebsocket(robot_ip)
    WS->>RB: WebSocket 连接
    MA->>R: listen_dialogue(push2web)，回调 POST 127.0.0.1:50001/add_message
    MA->>R: run()，录音与 VAD 线程写入 vad_queue
    loop _duplex 循环
        U->>R: 语音经录音与 VAD 线程（vad_queue）
        R->>ASR: recognizer(voice_data)
        ASR-->>R: text
        Note over R: 唤醒词 / 退出词判断，有效文本则 executor.submit(chat, text)
        R->>R: chat 与 chat_tool_by_content
        alt 规则匹配 preprocess_asr_result 命中
            R->>R: 得到 functions 列表
        else 未命中且 use_llm
            R->>LLM: response_call(dialogue, get_functions())
            LLM-->>R: 流式 chunk，含 tools_calls（TronAction 参数）
        end
        loop 每个 function
            R->>TM: call_tools → tool_call(name, args)
            TM->>TA: TronAction(**args)，ToolType.WAIT 同步调用
            TA->>FL: flow.run({action_commands})
            FL->>FL: CallLLM.post 校验 stand / sit / twist / height / light
            FL->>WS: 分派到 CallSit / CallStand / CallTwist / CallHeight / CallLight
            WS->>RB: send_request（request_sitdown、request_walk_mode、request_twist 等）
        end
        R->>TTS: speak_and_play(response)，to_tts 后 player.play
        Note over R: 用户文本与回复经 callback(push2web) 发往 Web 面板
    end
```

图依据：limxdynamics/tron1-agent@bf564fc，main.py、start.sh、config/config.yaml、bailing/robot.py、plugins/task_manager.py、plugins/functions/tron_action.py、third_party/tron/action/base_action.py、third_party/tron/communication/tron_websocket.py、third_party/tron/communication/base_websocket.py、third_party/tron/flow/base.py、third_party/tron/app/pocket.py、bailing/llm.py

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [tron1-agent 源码归档](../../sources/repos/tron1-agent.md)（<https://github.com/limxdynamics/tron1-agent>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：limxdynamics/tron1-agent@bf564fc，读过 main.py、start.sh、config/config.yaml、bailing/robot.py（关键调用段）、plugins/task_manager.py、plugins/functions/tron_action.py、third_party/tron/action/base_action.py、third_party/tron/communication/tron_websocket.py；对 bailing/llm.py、third_party/tron/communication/base_websocket.py、third_party/tron/flow/base.py、third_party/tron/app/pocket.py 仅检索函数定义；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [逐际动力 官方组织](https://github.com/limxdynamics)
