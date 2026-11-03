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
  - ../../sources/repos/noetix_sdk_bumi.md
summary: "松延动力 开源项目 noetix_sdk_bumi（SDK/驱动）：通过DDS提供Bumi机器人的高层与低层控制示例，支持上层应用和本体联调。…"
institutions:
  - noetix
---

# noetix_sdk_bumi

## 一句话定义

**noetix_sdk_bumi** 是 [松延动力](https://github.com/Noetix-Robotics) 公开的 **SDK/驱动** 开源项目：通过DDS提供Bumi机器人的高层与低层控制示例，支持上层应用和本体联调。

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

1. 从官方 GitHub/Gitee 组织检索 `noetix_sdk_bumi` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`./build.sh` 后在仓库根目录运行 `build/example_policy_walk`（ONNX 策略走路；还有 `example_high` 高层指令、`example_low` 低层关节、`example_media`，以及 pybind 模块 `highcontrol_py`/`lowcontrol_py`/`mediacontrol_py` 与 `examples/py/test_*.py`）。SDK 本体是预编译静态库 `lib/<arch>/libnoetix_sdk.a`（DDS 通信在库内，仅有头文件 `include/*.h`，库内部不可读），下图只画示例源码可见的调用。

```mermaid
sequenceDiagram
    participant M as example_policy_walk.cpp main
    participant LC as LowController (lowcontroller.h，libnoetix_sdk.a)
    participant HW as 机器人 DDS 话题 (config/dds.xml)
    participant JS as aolion_driver 手柄 /dev/input/js0
    participant ORT as onnxruntime 会话 policy/policy.onnx
    M->>M: setenv CYCLONEDDS_URI=config/dds.xml
    M->>LC: LowController::Instance()->init()
    M->>ORT: loadModel(policy.onnx) 并 getmodelparam 读取模型参数
    M->>JS: init() 中 aoliondriver.init + subscribe_robot_hardware_status 回调
    HW-->>M: 回调写入 latest_hw_status (motor_data, imu_data)
    loop 每 2ms 的 process()
        M->>JS: getremotedata 读摇杆与按键
        M->>M: 按键切换 WorkMode DEFAULT/LIE/STAND/USERMODE
        alt USERMODE
            M->>M: updateStateEstimation 按 jointNames 经 getJointsIndex 读关节与 IMU
            M->>M: 摔倒保护 projectedGravity(2) >= -0.3 则回 DEFAULT
            M->>ORT: 每 decimation 步 computeObservation 后 computeActions 执行 Run
            M->>M: 动作裁剪，pos_des = action*action_scale + default_joint_pos
            M->>LC: set_joint(array of 21 MotorCmd：pos, kp, kd)
            LC->>HW: DDS 发送关节指令
        else STAND/LIE
            M->>LC: handleStandMode/handleLieMode 内插值后 set_joint
        end
    end
```

图依据：Noetix-Robotics/noetix_sdk_bumi@ff0bc6a，读过 README.md、build.sh、CMakeLists.txt、include/{lowcontroller,highcontroller}.h、examples/cpp/{example_policy_walk,example_high,example_low}.cpp（example_policy_walk.cpp 读了 main、init、process、updateStateEstimation、computeActions、handleUserMode 与 loadModel 开头，未通读 818 行）；libnoetix_sdk.a 内部（DDS 话题、线程 send_thread_func）为二进制，未读；handleStandMode/handleLieMode 内的 set_joint 调用为按函数名推断，未逐行核对。

## 关联页面

- [Scalabot / HERON-World Model](./scalabot-heron-world-model.md)（松延动力 2026-09 发布的具身大脑子品牌）
- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [noetix_sdk_bumi 源码归档](../../sources/repos/noetix_sdk_bumi.md)（<https://github.com/Noetix-Robotics/noetix_sdk_bumi>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Noetix-Robotics/noetix_sdk_bumi@ff0bc6a，读过 README.md、build.sh、CMakeLists.txt、include/lowcontroller.h、include/highcontroller.h、examples/cpp/example_policy_walk.cpp、example_high.cpp、example_low.cpp；时序图依据以上文件，未运行代码；预编译库 libnoetix_sdk.a 内部未读。

## 推荐继续阅读

- [松延动力 官方组织](https://github.com/Noetix-Robotics)
