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
  - ../../sources/repos/booster_deploy.md
summary: "加速进化 开源项目 booster_deploy（部署运行时）：策略部署运行时：加载训练策略、下发关节命令…"
institutions:
  - booster
---

# booster_deploy

## 一句话定义

**booster_deploy** 是 [加速进化](https://github.com/BoosterRobotics) 公开的 **部署运行时** 开源项目：策略部署运行时：加载训练策略、下发关节命令

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
| 机构 | 加速进化 |
| 类别 | 部署运行时 |
| 官方组织 | https://github.com/BoosterRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `booster_deploy` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/deploy.py --task <TASK_NAME> --mujoco`（sim2sim）；机器人端为 `python scripts/deploy.py --task <TASK_NAME>`（sim2real，需 ROS 2 与 `booster_interface`）。

```mermaid
sequenceDiagram
    participant U as scripts/deploy.py
    participant REG as utils/registry.py + tasks/*
    participant MJ as MujocoController (mujoco_controller.py)
    participant PT as BoosterRobotPortal (booster_robot_controller.py)
    participant BC as BoosterRobotController（mp.Process 子进程）
    participant POL as LocomotionPolicy (tasks/locomotion/locomotion.py)
    participant RUN as policy_runner.create_policy_runner
    participant BOT as 机器人 /low_state、joint_ctrl、booster_rpc_service

    U->>REG: pkgutil.walk_packages 导入 tasks，各任务 register_task
    U->>REG: get_task(--task) 得 ControllerCfg
    alt --mujoco
        U->>MJ: MujocoController(task_cfg).run()
        MJ->>POL: BaseController.__init__ 构造 policy
        POL->>RUN: 按后缀选 TorchScript 或 ONNX Runtime
        loop viewer 运行中
            MJ->>MJ: update_state 读 mj_data
            MJ->>POL: policy_step -> inference
            POL->>RUN: 观测历史 -> 动作
            MJ->>MJ: ctrl_step: PD 力矩 + decimation 次 mj_step
        end
    else 真机
        U->>PT: with BoosterRobotPortal(task_cfg) as portal: portal.run()
        PT->>BOT: 创建 low_cmd 发布器、订阅 /low_state
        PT->>BOT: 等遥控器 X 与首帧 low_state，先发一帧保持指令，再经 booster_rpc_service 切 Custom 模式
        PT->>BC: mp.Process(inference_process_func)
        BC->>POL: BoosterRobotController -> 构造 policy
        POL->>RUN: create_policy_runner
        loop 每个 policy_dt
            BOT-->>PT: _low_state_handler 写 synced_state / synced_command
            BC->>PT: update_state 读 synced_state
            BC->>POL: policy_step -> inference
            BC->>BOT: ctrl_step 填 motor_cmd 的 q/kp/kd 并 publish
        end
        PT->>BOT: 退出时 RPC 切 exit_mode（默认 walking，异常 damping）
    end
```

图依据：BoosterRobotics/booster_deploy@7bb1462，读过 `README.md`、`scripts/deploy.py`、`booster_deploy/controllers/{base_controller,mujoco_controller,booster_robot_controller}.py`、`booster_deploy/utils/{policy_runner,registry}.py`、`tasks/locomotion/locomotion.py`（`beyond_mimic` 任务的 `register_task` 仅见 grep，未细读）。真机路径中 `prepare_mode="walking"` 的"先跑零指令行走策略、按 A 后切任务策略"分支未画。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [booster_deploy 源码归档](../../sources/repos/booster_deploy.md)（<https://github.com/BoosterRobotics/booster_deploy>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：BoosterRobotics/booster_deploy@7bb1462，读过 README.md、scripts/deploy.py、booster_deploy/controllers/{base_controller,mujoco_controller,booster_robot_controller}.py、booster_deploy/utils/{policy_runner,registry}.py、tasks/locomotion/locomotion.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [加速进化 官方组织](https://github.com/BoosterRobotics)
