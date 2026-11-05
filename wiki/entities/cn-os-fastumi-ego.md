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
  - ../../sources/repos/fastumi_ego.md
summary: "鹿明机器人 开源项目 FastUMI_Ego（数据集/Benchmark）：提供FastUMI第一视角数据采集入口，记录人类操作过程以服务机器人模仿学习数据构建。…"
institutions:
  - lumos
---

# FastUMI_Ego

## 一句话定义

**FastUMI_Ego** 是 [鹿明机器人](https://github.com/LumosRobot) 公开的 **数据集/Benchmark** 开源项目：提供FastUMI第一视角数据采集入口，记录人类操作过程以服务机器人模仿学习数据构建。

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
| 机构 | 鹿明机器人 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/LumosRobot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `FastUMI_Ego` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（Ubuntu 20.04 + ROS1 Noetic + FastUMI Hardware SDK 的 `xv_sdk`）：先 `bash Start_Process/unified_launcher.sh` 启动相机 ROS 节点，再 `bash Start_Process/pairing_process.sh` 配对写出 `camera_config.json`，最后 `python3 Data_Collector/live_hand_pose_2process_new.py` 开始录制与手部位姿估计。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant UL as unified_launcher.sh
    participant PR as pairing_process.sh
    participant ROS as ROS 话题 /xv_sdk/<sn>/...
    participant M as live_hand_pose_2process_new.py main
    participant N as RosHandPoseRecorderNode
    participant IR as RosImageRecorder
    participant HUB as LatestFrameHub
    participant MP as MediaPipe HandLandmarker

    U->>UL: 启动相机 ROS 节点（查找 catkin_ws，source ROS 环境）
    U->>PR: 配对
    PR->>ROS: rostopic list 提取 <serial>
    PR->>PR: 写 Start_Process/camera_config.json（device_sn、topic_prefix）
    U->>M: python3 live_hand_pose_2process_new.py
    M->>M: rospy.init_node，resolve_device_sn 读 camera_config.json
    M->>N: RosHandPoseRecorderNode()（resolve_task_setup、setup_task_session）
    N->>ROS: subscribe_topics：camera_info、color image、(TOF) rgbd、slam pose
    ROS-->>IR: on_color_image 回调
    IR->>IR: cv2.VideoWriter 写 mp4，记录帧时间
    IR->>HUB: publish(FramePacket)，只保留最新帧
    loop N.run() 主循环
        N->>HUB: get_latest_if_new(last_seq)
        N->>MP: detect_for_video(mp_image, mp_timestamp_ms)
        MP-->>N: hand_landmarks / hand_world_landmarks
        N->>N: solve_one_hand_pose（PnP + 逐关节坐标系）
        N->>N: write_hand_pose_to_joint_files 写每关节 CSV（left/right）
    end
    ROS-->>N: on_slam_pose → 写 SLAM 位姿 CSV
    N->>N: 退出时 cleanup 关闭写文件
```

图依据：LumosRobot/FastUMI_Ego@9caafcb，读过 README.md（目录与流程部分）、Start_Process/unified_launcher.sh、Start_Process/pairing_process.sh、Data_Collector/live_hand_pose_2process_new.py（类与函数清单、subscribe_topics、process_latest_frame、main；RosImageRecorder 内部与按键控制未逐行读）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [FastUMI_Ego 源码归档](../../sources/repos/fastumi_ego.md)（<https://github.com/LumosRobot/FastUMI_Ego>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LumosRobot/FastUMI_Ego@9caafcb，读过 README.md（目录与流程）、Start_Process/unified_launcher.sh、Start_Process/pairing_process.sh、Data_Collector/live_hand_pose_2process_new.py（主要部分，RosImageRecorder 内部未逐行读）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [鹿明机器人 官方组织](https://github.com/LumosRobot)
