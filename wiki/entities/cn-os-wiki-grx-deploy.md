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
  - ../../sources/repos/wiki-grx-deploy.md
summary: "傅利叶智能 开源项目 Wiki-GRx-Deploy（部署运行时）：策略部署运行时：加载训练策略、下发关节命令…"
institutions:
  - fourier
---

# Wiki-GRx-Deploy

## 一句话定义

**Wiki-GRx-Deploy** 是 [傅利叶智能](https://github.com/FFTAI) 公开的 **部署运行时** 开源项目：策略部署运行时：加载训练策略、下发关节命令

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
| 类别 | 部署运行时 |
| 官方组织 | https://github.com/FFTAI |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Wiki-GRx-Deploy` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 源码运行时序图

入口：`python developer_radian/demo_walk.py`（需先安装 PyPI 包 `fourier-grx`，且 4.0.0 及以上；4.0.0 以下用 `developer/demo_walk.py`）。本仓库只有示例脚本和策略权重，`fourier_grx` SDK 本体不在仓库内。

```mermaid
sequenceDiagram
    participant M as developer_radian/demo_walk.py (demo_task)
    participant CS as fourier_grx.sdk.developer.ControlSystem（外部包）
    participant PT as torch.jit.load(policy_jit_walk.pt)
    participant S as ischedule.schedule / run_loop
    participant A as algorithm()
    participant R as N1 机器人（经 fourier-grx 底层）

    M->>CS: ControlSystem()
    M->>CS: developer_mode(servo_on=True, control_frequency=100)
    CS->>R: 切换开发者模式，伺服使能
    M->>CS: get_info() 打印版本
    M->>PT: 加载 policy_jit_walk.pt（CPU）
    M->>S: schedule(algorithm, interval=1/50 s)；run_loop()
    loop 每 20ms（50Hz）
        S->>A: algorithm()
        A->>CS: robot_control_loop_get_state()
        CS-->>A: imu_quat、imu_angular_velocity、joint_position、joint_velocity
        A->>A: 取 13 个腿+腰关节，构造观测 obs_buf（命令、角速度、投影重力、关节偏差、速度*0.1、上次动作），堆叠 5 帧
        A->>PT: policy_model(obs_buf_stack)
        PT-->>A: 动作，按 action_clip_min/max 裁剪，加 default_joint_position
        A->>CS: robot_control_loop_set_control({control_mode: PD, pd_control_kp, pd_control_kd, position})
        CS->>R: 下发关节 PD 目标
    end
```

图依据：FFTAI/Wiki-GRx-Deploy@4de1420，读过 `README.md`、`developer_radian/README.md`、`developer_radian/demo_walk.py`（全文）、`developer_radian/config_N1__developer.yaml`（头部）、`user/demo_walk.py` 与 `pubsub_radian/demo_walk.py`（仅 import 与头部注释）。`fourier_grx` 内部（`ControlSystem` 如何与硬件 / Zenoh 通信）未读，图中 CS 到机器人一段为按 API 命名与 yaml 中 `zenoh` / `hardware_layer` 字段的推断。`user/` 与 `pubsub_radian/` 两套接口的调用流程未画。

## 参考来源

- [Wiki-GRx-Deploy 源码归档](../../sources/repos/wiki-grx-deploy.md)（<https://github.com/FFTAI/Wiki-GRx-Deploy>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：FFTAI/Wiki-GRx-Deploy@4de1420，读过 README.md、developer_radian/README.md、developer_radian/demo_walk.py、developer_radian/config_N1__developer.yaml（头部）、user/demo_walk.py 与 pubsub_radian/demo_walk.py（头部）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傅利叶智能 官方组织](https://github.com/FFTAI)
