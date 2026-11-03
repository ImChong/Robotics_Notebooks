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
  - ../../sources/repos/efmnode.md
summary: "星海图 开源项目 EFMNode（部署运行时）：策略部署运行时：加载训练策略、下发关节命令…"
institutions:
  - galaxea
---

# EFMNode

## 一句话定义

**EFMNode** 是 [星海图](https://github.com/OpenGalaxea) 公开的 **部署运行时** 开源项目：策略部署运行时：加载训练策略、下发关节命令

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
| 机构 | 星海图 |
| 类别 | 部署运行时 |
| 官方组织 | https://github.com/OpenGalaxea |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `EFMNode` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`./scripts/run.sh`（设置环境变量后执行 `python3 run.py [--model-path <ckpt_dir>]`；`run.py` 读取 `config.toml` 或 `<model_path>/efmnode.toml`）。模型网络结构由 `GalaxeaFM`（`galaxea_fm`）包提供，本仓库只负责 ROS2 收发、预处理/后处理与调度。

```mermaid
sequenceDiagram
    participant R as run.py
    participant S as scheduler/scheduler.py Scheduler
    participant B as core/communication/ros2_bridge.py Ros2Bridge
    participant IM as scheduler/instruction/instruction.py InstructionManager
    participant P as core/processor (create_processor)
    participant E as core/inference (PyTorchEngine / TensorRTEngine)
    participant TM as scheduler/trajectory/manager.py TrajectoryManager
    R->>R: load_config(--model-path)
    R->>S: Scheduler(config).run()
    S->>S: _setup_all：processor、trajectory_manager、instruction_manager、ros2_bridge、inference_engine
    S->>P: initialize(ckpt_dir/dataset_stats.json)
    S->>E: create_inference_engine(use_trt) 并 load_model()
    S->>B: register_subscription(String, hs/vlm_out2vla, 指令回调)
    loop while ros2_bridge.is_running()
        S->>B: gather_obs()
        B-->>S: obs_time, obs（图像、关节状态等，按时间戳匹配）
        S->>IM: get_instruction(obs)
        alt RESET
            S->>B: reset()
        else CONTINUE
            S->>P: preprocess(obs)
            S->>E: predict_action(batch)
            E-->>S: batch["action"], batch["proprio"]
            S->>P: postprocess(batch)
            P-->>S: actions
        end
        alt step_mode == sync
            S->>B: publish_action 逐步发布 action_steps 个动作，按 control_frequency 休眠
        else step_mode == async
            S->>TM: add_actions(actions, obs_time)
            B->>TM: 定时回调 _async_publish → get_action(now)
            B->>B: publish_action(action)
        end
    end
```

图依据：OpenGalaxea/EFMNode@154b69c，读过 `README.md`、`run.py`、`scripts/run.sh`、`scripts/launch_server.sh`、`config.toml`、`scheduler/scheduler.py`、`core/inference/factory.py`、`core/inference/pytorch_engine.py`（函数骨架）、`core/communication/ros2_bridge.py`（函数骨架）、`serving/policy_server.py`（函数骨架）；`processor`、`trajectory/manager.py`、`instruction.py` 内部未逐行读。`serving/policy_server.py` 的 WebSocket 服务路径（`scripts/launch_server.sh`）未画入图中。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [EFMNode 源码归档](../../sources/repos/efmnode.md)（<https://github.com/OpenGalaxea/EFMNode>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：OpenGalaxea/EFMNode@154b69c，读过 README.md、run.py、scripts/run.sh、scripts/launch_server.sh、config.toml、scheduler/scheduler.py、core/inference/factory.py、core/inference/pytorch_engine.py、core/communication/ros2_bridge.py、serving/policy_server.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [星海图 官方组织](https://github.com/OpenGalaxea)
