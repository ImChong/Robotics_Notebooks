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
  - ../../sources/repos/abot-claw.md
summary: "阿里巴巴 开源项目 ABot-Claw（VLA/操作模型）：统一 VLN、VLA、WAM 与视觉记忆的持续协作机器人框架…"
institutions:
  - alibaba
---

# ABot-Claw

## 一句话定义

**ABot-Claw** 是 [阿里巴巴](https://github.com/amap-cvlab) 公开的 **VLA/操作模型** 开源项目：统一 VLN、VLA、WAM 与视觉记忆的持续协作机器人框架

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第四层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 阿里巴巴 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/amap-cvlab、https://github.com/alibaba-damo-academy |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `ABot-Claw` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

范围说明：本图只覆盖机器人侧 agent server（`robot_layer/arm_piper/agent_server`）的代码执行链路。OpenClaw 侧的技能与服务侧（`service_layer`）只在图中以外部 HTTP 服务出现，未逐一展开。

入口：`cd robot_layer/arm_piper/agent_server && python3 server.py --host 0.0.0.0 --port 8888`（先 `pip3 install -r requirements.txt`；`server.py` 的 `main()` 支持 `--dry-run`，不连接真机）

```mermaid
sequenceDiagram
    participant OC as OpenClaw 代理<br/>skills 调用方
    participant API as server.py<br/>build_app / FastAPI
    participant CR as routes/code_routes.py<br/>POST /execute
    participant LM as lease.py<br/>LeaseManager
    participant EX as code_executor.py<br/>CodeExecutor
    participant PY as 子进程 /usr/bin/python3<br/>生成的临时脚本
    participant SDK as robot_sdk<br/>PiperRobotEnv / YoloSDK / GraspSDK / MemorySDK
    participant SVC as 外部 HTTP 服务<br/>config.yaml 中的 url

    OC->>API: POST /execute（X-Lease-Id 与 code）
    API->>CR: execute_code
    CR->>LM: validate_lease(lease_id)
    alt 租约无效或已过期
        CR-->>OC: 403 Invalid or expired lease
    else 租约有效
        CR->>LM: record_command()
        CR->>EX: is_running 检查，然后 validate_code（AST 检查，非安全沙箱）
        alt 校验失败
            CR-->>OC: 返回校验错误
        else 校验通过
            CR->>CR: 创建后台任务 run_code（recorder.start）
            CR-->>OC: CodeExecuteResponse（execution_id，立即返回）
            CR->>EX: execute(code, timeout, lease_id, holder)
            EX->>EX: _create_temp_file：脚本头部注入 robot_sdk 路径与 SDK 初始化
            EX->>PY: subprocess.Popen(/usr/bin/python3 -u 临时脚本)
            PY->>SDK: PiperRobotEnv()，YoloSDK().start()，GraspSDK().start()，MemorySDK()
            SDK->>SVC: requests.post（yolo.url、grasp.url）
            SVC-->>SDK: 检测或抓取候选结果
            SDK->>SVC: 空间记忆 HTTP 请求（spatial_memory.url）
            SVC-->>SDK: 记忆查询结果
            SDK-->>PY: 机械臂与相机话题数据（ROS）
            PY-->>EX: stdout 与 stderr 流
            EX-->>CR: ExecutionResult（状态写入 get_last_result）
        end
    end
```

图依据：amap-cvlab/ABot-Claw@39c9ffc，读过 `README.md`、`robot_layer/arm_piper/agent_server/server.py`（`build_app`、`main`、路由注册）、`robot_layer/arm_piper/agent_server/routes/code_routes.py`（`/execute`、`/validate`、后台任务）、`robot_layer/arm_piper/agent_server/code_executor.py`（`CodeValidator`、`execute`、`_create_temp_file` 的脚本模板）、`robot_layer/arm_piper/agent_server/lease.py`（`validate_lease`、`record_command`）、`robot_layer/arm_piper/agent_server/robot_sdk/yolo_sdk.py`、`robot_layer/arm_piper/agent_server/robot_sdk/grasp_sdk.py`（HTTP 请求）、`robot_layer/arm_piper/agent_server/robot_sdk/memory_sdk.py`（`base_url`）、`robot_layer/arm_piper/agent_server/robot_sdk/config.yaml`。`PiperRobotEnv` 的 ROS 细节、`MemorySDK` 的具体查询接口、OpenClaw 技能与 `service_layer` 服务未逐行核对；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [ABot-Claw 源码归档](../../sources/repos/abot-claw.md)（<https://github.com/amap-cvlab/ABot-Claw>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：amap-cvlab/ABot-Claw@39c9ffc，读过 README.md、robot_layer/arm_piper/agent_server/server.py、robot_layer/arm_piper/agent_server/routes/code_routes.py、robot_layer/arm_piper/agent_server/code_executor.py、robot_layer/arm_piper/agent_server/lease.py、robot_layer/arm_piper/agent_server/robot_sdk/yolo_sdk.py、robot_layer/arm_piper/agent_server/robot_sdk/grasp_sdk.py、robot_layer/arm_piper/agent_server/robot_sdk/memory_sdk.py、robot_layer/arm_piper/agent_server/robot_sdk/config.yaml；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [阿里巴巴 官方组织](https://github.com/amap-cvlab)
