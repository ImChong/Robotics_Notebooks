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
  - ../../sources/repos/roban_model_and_sim.md
summary: "乐聚机器人 开源项目 roban_model_and_sim（本体模型资产）：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证…"
institutions:
  - leju
---

# roban_model_and_sim

## 一句话定义

**roban_model_and_sim** 是 [乐聚机器人](https://github.com/LejuRobotics) 公开的 **本体模型资产** 开源项目：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

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
| 机构 | 乐聚机器人 |
| 类别 | 本体模型资产 |
| 官方组织 | https://github.com/LejuRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `roban_model_and_sim` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：仓库只有 `taskNN_*.py` 比赛任务脚本（无 README、无模型/URDF 文件），各脚本作为 ROS 节点直接运行，如 `python task02_press_button.py`（需要外部包 `lss_roban`，不在本仓库内；运行在仿真或真机的 ROS 环境，脚本订阅的是 `/sim/camera/D435/colorImage`）。下图以 `task02_press_button.py`（行走 + 视觉按按钮）为例；其它 task 脚本的结构类似但未逐一画出，`task09_upstair_and_down.py` 则是不带视觉、仅用 `linearTrajMove` 串联上下楼梯动作。

```mermaid
sequenceDiagram
    autonumber
    participant M as task02_press_button.py __main__/main()
    participant L as lss_roban.motion_control（外部包，import *）
    participant CB as D435Colorcallback（订阅回调）
    participant TT as TrackingThread（4Hz 线程）
    participant R as ROS 话题 / 机器人

    M->>R: rospy.init_node + on_shutdown(rosShutdownHook)
    M->>R: Subscriber(/sim/camera/D435/colorImage)
    R-->>CB: Image → HSV 红/粉阈值 + 轮廓，更新 x,y,w,h、mask、see_button
    M->>L: setBodyhubNoStand(1), setArmMode(0), walkingInit()
    M->>TT: start()（pause/over 两个 threading.Event）
    M->>L: UpdateTargets()
    loop 直到 ikRightArm 对按钮前后两点都有解
        M->>M: getButtonPos()（pause.set 暂停追踪）
        M->>L: getCoordinatesFromDepthImage(w_x, w_y)
        TT->>L: visualTrack(button_pos)（pause 未置位时）
        M->>L: ikRightArm(button_pos ± 偏移)
        M->>L: WalkingThread(dx, dy, 0).start()
        M->>L: bodyMoveTo(right_arm=[arm_prepare], count=50)
    end
    M->>L: linearTrajMoveTo(right_arm=…) 三次：前探→按下→退回
    M->>L: 退出时 rosShutdownHook: setBodyhubNoStand(0), setArmMode(1), resetBodyhub()
```

图依据：LejuRobotics/roban_model_and_sim@71c51b9，读过 task01_press_button.py、task02_press_button.py、task09_upstair_and_down.py（其余 task 脚本未读）。`lss_roban.motion_control` 内部实现不在仓库，图中仅画出脚本对其函数的调用。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [roban_model_and_sim 源码归档](../../sources/repos/roban_model_and_sim.md)（<https://github.com/LejuRobotics/roban_model_and_sim>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LejuRobotics/roban_model_and_sim@71c51b9，读过 task01_press_button.py、task02_press_button.py、task09_upstair_and_down.py（仓库仅含 13 个 taskNN_*.py，无 README / 模型文件）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [乐聚机器人 官方组织](https://github.com/LejuRobotics)
