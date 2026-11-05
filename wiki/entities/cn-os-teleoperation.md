---
type: entity
tags: [repo, china-embodied-opensource, open-source, project, realman]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/teleoperation.md
summary: "傅利叶智能 开源项目 teleoperation（遥操作与数据采集）：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据…"
institutions:
  - fourier
---

# teleoperation

## 一句话定义

**teleoperation** 是 [傅利叶智能](https://github.com/FFTAI) 公开的 **遥操作与数据采集** 开源项目：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

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
| 机构 | 傅利叶智能 |
| 类别 | 遥操作与数据采集 |
| 官方组织 | https://github.com/FFTAI |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `teleoperation` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python -m teleoperation --config-name teleop robot=gr1t1_legacy hand=fourier_dexpilot_dhx camera=oak_97 robot.visualize=true mocap=avp sim=true`（README 示例；`src/teleoperation/__main__.py` 调 `main.main()`，Hydra 读 `configs/teleop.yaml`）。`sim=false` 时才接真机。

```mermaid
sequenceDiagram
    participant H as VR 头显（AVP / Quest，WebXR）
    participant TV as OpenTeleVision (television.py，基于 Vuer)
    participant M as main.main() (main.py)
    participant R as TeleopRobot (player.py，继承 DexRobot/IKRobot)
    participant P as VuerPreprocessor (preprocess.py)
    participant U as Upsampler 线程 (upsampler.py)
    participant A as GR1Robot 适配器 (adapter/robots/grx.py)
    participant C as fourier_grx_client.RobotClient
    M->>R: TeleopRobot(cfg)
    R->>R: hydra 实例化 camera，创建 OpenTeleVision 与 VuerPreprocessor
    alt sim=false
        R->>A: hydra 实例化 cfg.robot.instance，connect()
        R->>U: Upsampler(client, target_hz)
    else sim=true
        R->>R: DummyRobot / DummyDexHand
    end
    H->>TV: on_cam_move / on_hand_move 事件（头部与手腕矩阵、手部关键点）
    TV-->>R: 首次收到手部数据置 connected=True
    loop 每帧，频率 cfg.frequency
        M->>R: step()
        R->>P: processor.process(tv) 坐标系 y-up 转 z-up，得到 head/wrist/hand 矩阵
        R->>R: hand_retarget.retarget(left_hand_mat, right_hand_mat)
        R-->>M: head_mat, 左右 wrist_mat, 左右手 qpos
        M->>R: solve(left_wrist, right_wrist, head, dt) 调 pink.solve_ik 做差分 IK
        M->>M: FSM 状态推进（按空格：CALIBRATING 到 ENGAGED）
        M->>R: control_hands(left_qpos, right_qpos)
        M->>R: control_joints() 后 upsampler.put(qpos)
        U->>A: _send_command 调 command_joints(command)
        A->>C: client.move_joints(ControlGroup.ALL, positions)
    end
    Note over M,R: cfg.recording.enabled 时，COLLECTING 状态下 data_dict.add_action / add_state，回合结束 recording.save_episode
```

图依据：FFTAI/teleoperation@f90a64a，读过 README.md、src/teleoperation/__main__.py、main.py、player.py（TeleopRobot）、television.py、preprocess.py（process）、retarget/robot.py（solve）、upsampler.py、adapter/robots/grx.py。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [teleoperation 源码归档](../../sources/repos/teleoperation.md)（<https://github.com/FFTAI/teleoperation>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：FFTAI/teleoperation@f90a64a，读过 README.md、src/teleoperation/__main__.py、main.py、player.py、television.py、preprocess.py、retarget/robot.py、upsampler.py、adapter/robots/grx.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傅利叶智能 官方组织](https://github.com/FFTAI)
