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
  - ../../sources/repos/noetix_sdk_n2.md
summary: "松延动力 开源项目 noetix_sdk_n2（SDK/驱动）：提供N2人形机器人状态读取与控制接口，连接策略运行时和真实本体。…"
institutions:
  - noetix
---

# noetix_sdk_n2

## 一句话定义

**noetix_sdk_n2** 是 [松延动力](https://github.com/Noetix-Robotics) 公开的 **SDK/驱动** 开源项目：提供N2人形机器人状态读取与控制接口，连接策略运行时和真实本体。

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
| 机构 | 松延动力 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/Noetix-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `noetix_sdk_n2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`./build.sh` 后在仓库根目录运行 `build/example_low`（ONNX 策略走路，读 `policy/policy_user.onnx`）或 `build/example_high`（高层指令）；另有 pybind 模块 `highcontrol_py` / `lowcontrol_py` 与 `examples/py/test_*.py`（含 `test_policy.py`，未读）。SDK 本体是预编译静态库 `lib/<arch>/libnoetix_n2_sdk.a`（DDS 在库内，仓库只有头文件），库内部不可读，下图只画示例源码可见的调用。

```mermaid
sequenceDiagram
    participant M as examples/cpp/example_low.cpp main
    participant LC as LowController (lowcontroller.h，libnoetix_n2_sdk.a)
    participant HW as 机器人 DDS 话题 (config/dds.xml)
    participant ORT as onnxruntime 会话 policy_user.onnx
    participant H as examples/cpp/example_high.cpp
    participant HC as HighController (highcontroller.h)
    M->>M: setenv CYCLONEDDS_URI=config/dds.xml
    M->>LC: LowController::Instance()->init()
    M->>ORT: loadModel(policy_user.onnx)
    M->>LC: init() 中 subscribe_robot_hardware_status(回调存 latest_hw_status)
    HW-->>M: 回调更新 motor_data / imu_data / remote_data
    loop 每 2ms 的 process()
        M->>M: remote_data 的摇杆与按键切换工作模式
        alt 行走模式 handleWalkMode
            M->>M: updateStateEstimation 读关节与 IMU
            M->>ORT: 每 decimation 步 computeObservation 后 computeActions 执行 Run
            M->>M: 动作按 clipActions 裁剪
            M->>LC: set_joint(array of 18 MotorCmd)
            LC->>HW: DDS 发送关节指令
        else 站立/趴下/默认 handleStandMode/handleLieMode
            M->>LC: set_joint(插值后的 MotorCmd)
        end
    end
    H->>HC: Instance()->init()，subscribe_robot_hardware_status
    loop 主循环
        H->>HC: publish_cmd(x, y, yaw, ControlCmd, fileindex)
    end
```

图依据：Noetix-Robotics/noetix_sdk_n2@bdd7268，读过 README.md、CMakeLists.txt（add_executable/pybind 行）、include/{lowcontroller,highcontroller}.h、examples/cpp/example_low.cpp（main、init 的订阅段、handleWalkMode 开头、process 开头，以及 set_joint 调用位置的 grep）、examples/cpp/example_high.cpp（init/subscribe/publish_cmd 位置）；libnoetix_n2_sdk.a 内部为二进制未读；process 中模式切换细节、handleDefautMode 内行为、pybind 与 test_policy.py 未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [noetix_sdk_n2 源码归档](../../sources/repos/noetix_sdk_n2.md)（<https://github.com/Noetix-Robotics/noetix_sdk_n2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Noetix-Robotics/noetix_sdk_n2@bdd7268，读过 README.md、CMakeLists.txt、include/lowcontroller.h、include/highcontroller.h、examples/cpp/example_low.cpp、examples/cpp/example_high.cpp；时序图依据以上文件，未运行代码；预编译库内部未读。

## 推荐继续阅读

- [松延动力 官方组织](https://github.com/Noetix-Robotics)
