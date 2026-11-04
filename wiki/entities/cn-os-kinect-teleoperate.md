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
  - ../../sources/repos/kinect_teleoperate.md
summary: "宇树科技 开源项目 kinect_teleoperate（遥操作与数据采集）：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据…"
institutions:
  - unitree
---

# kinect_teleoperate

## 一句话定义

**kinect_teleoperate** 是 [宇树科技](https://github.com/unitreerobotics) 公开的 **遥操作与数据采集** 开源项目：遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

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
| 机构 | 宇树科技 |
| 类别 | 遥操作与数据采集 |
| 官方组织 | https://github.com/unitreerobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `kinect_teleoperate` 仓库并核对 README 许可与依赖。
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

入口：`./kinect_teleoperate`（在 `build/` 目录运行；`main()` 位于 `src/kinect_teleoperate_robot/main.cpp`，当前代码 `Real_Control=false`，仅驱动 MuJoCo 仿真，不向真机下发指令）

```mermaid
sequenceDiagram
    participant M as main.cpp: main()
    participant MI as main.cpp: Main_loop()
    participant K as Azure Kinect SDK (k4a / k4abt)
    participant MATH as math_tool.hpp
    participant CL as main.cpp: Control_loop (线程)
    participant RT as jointRetargeting.hpp
    participant SP as StartEndPoseDetector.hpp
    participant MJ as MuJoCo 3.1.5 (m / d)
    participant KR as main.cpp: KinectRender_loop (线程)
    participant MR as main.cpp: MujocoRender_loop (线程)

    M->>MJ: mj_loadXML(unitree_h1/mjcf/scene.xml), mj_makeData, mj_resetData
    M->>CL: std::thread control_thread
    M->>MR: std::thread mujocoRender_thread
    M->>MI: 主线程调用 Main_loop()
    MI->>K: k4a_device_open, k4a_device_start_cameras (NFOV_UNBINNED)
    MI->>K: k4a_device_get_calibration + k4abt_tracker_create (GPU CUDA)
    MI->>KR: std::thread KinectRender_thread
    loop 主循环 while s_isRunning
        MI->>K: k4a_device_get_capture (非阻塞)
        MI->>K: k4abt_tracker_enqueue_capture, k4abt_tracker_pop_result
        MI->>MI: 保存 globalBodyFrameForSkeleton (引用计数)
        MI->>MATH: ProcessNewSkeletonData: quaternion2Euler, calculateRelativeQuaternion
    end
    Note over MI,CL: 关节角经全局变量 ls_r / rs_r 等共享, 无同步原语
    loop 控制循环 while s_isRunning
        CL->>RT: LS / RS / LE / RE mappingCamera...2RobotRadians
        RT-->>CL: 机器人关节角
        CL->>SP: isStartEndPose(8 个关节角)
        SP-->>CL: state (START / END 翻转)
        alt state 为真
            CL->>MATH: MovingAverageFilter.update 平滑
            CL->>MJ: mj_step1, 写执行器 ctrl, mj_step2
        end
    end
    loop Kinect 渲染循环
        KR->>K: 读取 body skeleton, body index map, depth image
        KR->>KR: renderSkeletonAndPointCloud: AddJoint, AddBone, UpdatePointClouds
    end
    loop MuJoCo 渲染循环
        MR->>MJ: mjv_updateScene, mjr_render
        MR->>MR: glfwSwapBuffers, glfwPollEvents
    end
```

图依据：unitreerobotics/kinect_teleoperate@c520506，`README.md`、`src/kinect_teleoperate_robot/main.cpp`、`include/StartEndPoseDetector.hpp`、`include/jointRetargeting.hpp`、`include/math_tool.hpp`、`src/kinect_teleoperate_robot/CMakeLists.txt`。

## 参考来源

- [kinect_teleoperate 源码归档](../../sources/repos/kinect_teleoperate.md)（<https://github.com/unitreerobotics/kinect_teleoperate>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：unitreerobotics/kinect_teleoperate@c520506，读过 README.md、src/kinect_teleoperate_robot/main.cpp、include/StartEndPoseDetector.hpp、include/jointRetargeting.hpp、include/math_tool.hpp、src/kinect_teleoperate_robot/CMakeLists.txt；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [宇树科技 官方组织](https://github.com/unitreerobotics)
