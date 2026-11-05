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
  - ../../sources/repos/oglove_ros2_pkg.md
summary: "傲意科技 开源项目 oglove_ros2_pkg（SDK/驱动）：OGlove 手势数据手套 ROS2 包…"
institutions:
  - ohand
---

# oglove_ros2_pkg

## 一句话定义

**oglove_ros2_pkg** 是 [傲意科技](https://github.com/oymotion) 公开的 **SDK/驱动** 开源项目：OGlove 手势数据手套 ROS2 包

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
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/oymotion |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `oglove_ros2_pkg` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 run oglove_ros2 usb_glove_node`（setup.py 的 console_scripts 入口，对应 `scripts/usb_glove_ros2.py` 的 `main`；README 写作 `ros2 oglove_ros2 usb_glove_node`，此处按 ROS2 标准 `ros2 run` 书写）

仅画 USB 手套路线，BLE 手套（`ble_glove_node`）未画。

```mermaid
sequenceDiagram
    participant U as 用户终端
    participant N as DualGlovePublisher（usb_glove_publisher 节点）
    participant F as find_glove_ports()
    participant G as OGlove（每只手套）
    participant AL as asyncio 事件循环（独立线程）
    participant T as ROS2 定时器 0.01 s
    participant P as 话题发布者（左右手）
    U->>N: ros2 run oglove_ros2 usb_glove_node
    N->>F: find_glove_ports() 枚举候选串口
    loop 每个候选串口
        N->>G: serial.Serial(115200) 后构造 OGlove
    end
    N->>AL: 启动事件循环线程 run_forever
    N->>P: 创建 /left_usb_glove_publisher/glove_finger_data 与 /right_usb_glove_publisher/glove_finger_data
    N->>T: create_timer(0.01, timer_callback)
    N->>N: main 中临时事件循环 run_until_complete(calibrate_all_gloves())
    N->>G: 每只手套 start_calibration()，内部调用 get_data 读取标定数据
    N->>N: rclpy.spin 开始，定时器才会触发
    loop 每 10 ms
        T->>AL: run_coroutine_threadsafe(async_get_and_publish)
        AL->>G: get_position() 读串口并插值到 0-65535
        AL->>P: 按 left_or_right 发布 JointState，frame_id 为 left_glove 或 right_glove
    end
    U->>N: Ctrl+C，KeyboardInterrupt 后 destroy_node()
    N->>AL: 停止事件循环并 join 线程，关闭各串口
```

图依据：oymotion/oglove_ros2_pkg@d911e4f，读过 README.md、setup.py（entry_points 段）、scripts/usb_glove_ros2.py（OGlove 的 __init__、get_data、start_calibration、get_position，以及 find_glove_ports、DualGlovePublisher、main）。未运行代码。BLE 节点与 scripts/lib_gforce 未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [oglove_ros2_pkg 源码归档](../../sources/repos/oglove_ros2_pkg.md)（<https://github.com/oymotion/oglove_ros2_pkg>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：oymotion/oglove_ros2_pkg@d911e4f，读过 README.md、setup.py、scripts/usb_glove_ros2.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
