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
  - ../../sources/repos/atom-locomotion-deploy.md
summary: "越疆科技 开源项目 atom-locomotion-deploy（仿真环境）：把ATOM运动策略接入仿真或真机运行时，处理机器人状态、策略推理与控制命令输出。…"
institutions:
  - dobot
---

# atom-locomotion-deploy

## 一句话定义

**atom-locomotion-deploy** 是 [越疆科技](https://github.com/embodied-dobot) 公开的 **仿真环境** 开源项目：把ATOM运动策略接入仿真或真机运行时，处理机器人状态、策略推理与控制命令输出。

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
| 机构 | 越疆科技 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/embodied-dobot、https://github.com/Dobot-Arm |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `atom-locomotion-deploy` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`cd atom-locomotion-deploy && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && ./atom_deploy`（需先在 `dobot_atom_sdk` 目录下构建 DOBOT Atom SDK 与 ONNX Runtime，见 README 安装步骤）。

```mermaid
sequenceDiagram
    participant U as 用户 Ctrl+C 与遥控器
    participant M as src/atom_deploy.cpp<br/>main 与 AtomDeploy
    participant L as include/loop/loop.hpp<br/>LoopFunc
    participant F as include/robot_fsm/fsm.hpp<br/>AtomDeployFsm
    participant B as Atom::Bridge<br/>dobot_atom_sdk
    participant R as include/robot_model/atom.hpp<br/>ATOMModel
    participant O as ONNX Runtime
    U->>M: ./atom_deploy 构造 AtomDeploy
    M->>M: ReadYaml 读取 policy/config.yaml 的 atom 节点
    M->>F: make_shared AtomDeployFsm 初始状态 StatePassive
    M->>L: loop_control 周期 dt 即 500 Hz 回调 RobotControl
    M->>L: loop_rl 周期 inference_dt 即 100 Hz 回调 RunModel
    M->>L: start 两个循环线程
    loop 每个 500 Hz 控制周期 loop_control
        L->>M: RobotControl
        M->>B: GetState 读取 base joint arm 状态与遥控器按键
        M->>M: 按钮映射为 fsm_change A 默认位 B 初始位 R1U 模型 SELECT 被动
        M->>F: dispatch_event 对应事件或 EventTick
        F->>M: on_event 按当前状态写 leg_command 与 arm_command
        opt 首次进入 StateModel
            F->>M: InitRL 设置观测并构造模型
            M->>R: ATOMModel(params)
            R->>O: 创建 Session 加载 POLICY_DIR 下 model_name
        end
        F->>M: 从 output_dof_pos_queue 与 output_dof_vel_queue 取出 RL 输出写 q_ref
        M->>B: SetCommand 发送 fsm id 2 与 leg arm 指令
    end
    loop 每个 100 Hz 推理周期 loop_rl
        L->>M: RunModel
        opt rl_init_done 为真
            M->>M: Forward
            M->>R: compute_observation 由 base joint arm control 与 obs 组装
            R->>O: forward 执行 Session Run
            O-->>R: actions
            M->>M: clip_actions 后乘 action_scale 加 default_dof_pos
            M->>M: 推入 output_dof_pos_queue 与 output_dof_vel_queue
        end
    end
    U->>M: SIGINT 触发 signalHandler 置位 g_shutdown_requested
    M->>L: main 退出循环后析构 AtomDeploy 调用 shutdown
```

图依据：embodied-dobot/atom-locomotion-deploy@87ecbd5，读过 README.md（Build & Run 段）、src/atom_deploy.cpp、include/robot_fsm/fsm.hpp（状态与事件分发段）、include/robot_model/atom.hpp（ONNX 会话创建段）、include/loop/loop.hpp（LoopFunc 线程与 start/shutdown）、policy/config.yaml、CMakeLists.txt（POLICY_DIR 定义与链接库段）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [atom-locomotion-deploy 源码归档](../../sources/repos/atom-locomotion-deploy.md)（<https://github.com/embodied-dobot/atom-locomotion-deploy>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：embodied-dobot/atom-locomotion-deploy@87ecbd5，读过 README.md、src/atom_deploy.cpp、include/robot_fsm/fsm.hpp、include/robot_model/atom.hpp、include/loop/loop.hpp、policy/config.yaml、CMakeLists.txt；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [越疆科技 官方组织](https://github.com/embodied-dobot)
