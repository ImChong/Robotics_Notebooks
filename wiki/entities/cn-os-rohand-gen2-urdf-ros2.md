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
  - ../../sources/repos/rohand_gen2_urdf_ros2.md
summary: "傲意科技 开源项目 rohand_gen2_urdf_ros2（本体模型资产）：ROHand Gen2 灵巧手 URDF（ROS2）…"
institutions:
  - ohand
---

# rohand_gen2_urdf_ros2

## 一句话定义

**rohand_gen2_urdf_ros2** 是 [傲意科技](https://github.com/oymotion) 公开的 **本体模型资产** 开源项目：ROHand Gen2 灵巧手 URDF（ROS2）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第三层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 傲意科技 |
| 类别 | 本体模型资产 |
| 官方组织 | https://github.com/oymotion |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `rohand_gen2_urdf_ros2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch ap001 left_rviz2.launch.py`（ROS 2 Humble；右手入口为 `ap001/launch/right_rviz2.launch.py`，ap002 包下有同名文件，本图未逐一核对其内容）。

```mermaid
sequenceDiagram
    participant L as launch left_rviz2.launch.py
    participant GUI as rohand_joint_state_gui 滑块界面
    participant JSP as JointStatePublisher 外部包
    participant URDFN as rohand_urdf_node ROHandURDFNode
    participant FM as scripts/FingerMathURDF.py
    participant RSP as robot_state_publisher 外部节点
    participant RV as rviz2 外部节点

    L->>GUI: 启动 rohand_left_joint_state_gui 传入 urdf 路径
    L->>URDFN: 启动 rohand_left_urdf_node 命名空间 rohand_left
    L->>RSP: 启动 robot_state_publisher 加载 urdf
    L->>RV: 启动 rviz2 加载 rohand_left_urdf.rviz

    GUI->>JSP: JointStatePublisher(urdf_file) 读取自由关节
    loop 用户拖动滑块或随机回中
        GUI->>JSP: 写入滑块对应关节值 sliderUpdateCb
        JSP->>URDFN: 发布 JointState 到 rohand_left_urdf_node/joint_states
        URDFN->>URDFN: _joint_states_callback 按 *_slider_link 名称筛选
        URDFN->>FM: HAND_FingerPosToAngle(finger_id, position)
        FM-->>URDFN: 各指关节角
        URDFN->>URDFN: cal_joint_angle 组装 name 与 position
        URDFN->>RSP: 发布 JointState 到 /rohand_left/joint_states
        RSP->>RV: 经 /tf 提供各连杆位姿
    end
```

图依据：rohand_gen2_urdf_ros2@748c3ff，读过 ap001/launch/left_rviz2.launch.py、ap001/setup.py（entry_points 与 data_files）、ap001/scripts/rohand_urdf.py（全文）、ap001/scripts/rohand_joint_state_gui.py（`main` 与 `sliderUpdateCb` 段，并确认 import 自 `joint_state_publisher`）、ap001/scripts/FingerMathURDF.py（`HAND_FingerPosToAngle` 定义行）。未运行代码（未起 ROS 2 环境）；GUI 内部与 `JointStatePublisher` 的发布细节未读，图中仅按其 import 与 launch 重映射标注。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [rohand_gen2_urdf_ros2 源码归档](../../sources/repos/rohand_gen2_urdf_ros2.md)（<https://github.com/oymotion/rohand_gen2_urdf_ros2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：oymotion/rohand_gen2_urdf_ros2@748c3ff，读过 ap001/launch/left_rviz2.launch.py、ap001/setup.py、ap001/scripts/rohand_urdf.py、ap001/scripts/rohand_joint_state_gui.py、ap001/scripts/FingerMathURDF.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
