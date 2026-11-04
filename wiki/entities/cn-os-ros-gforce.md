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
  - ../../sources/repos/ros_gforce.md
summary: "傲意科技 开源项目 ros_gforce（SDK/驱动）：gForce Pro 肌电臂环 ROS 驱动…"
institutions:
  - ohand
---

# ros_gforce

## 一句话定义

**ros_gforce** 是 [傲意科技](https://github.com/oymotion) 公开的 **SDK/驱动** 开源项目：gForce Pro 肌电臂环 ROS 驱动

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

1. 从官方 GitHub/Gitee 组织检索 `ros_gforce` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python3 src/gforce_pub.py`（需先 `roscore`；`CMakeLists.txt` 中 `catkin_install_python` 处于注释状态，脚本未随包安装，且 shebang 为作者本机路径）

```mermaid
sequenceDiagram
    participant Main as gforce_pub.py __main__
    participant GF as GForceProfile（src/gforce.py）
    participant BLE as bluepy Scanner / Peripheral
    participant Dev as gForce 臂环（BLE）
    participant Ros as rospy 话题（gf_emg / gf_acc / gf_quat / gf_gyro / gf_euler）
    Main->>Main: rospy.get_param 读取 sample_rate / resolution / 各话题名
    loop 直到连接成功
        Main->>GF: GForceProfile()
        Main->>GF: scan(5.0)
        GF->>BLE: Scanner().scan(5.0)，按 GUID 过滤
        BLE->>Dev: BLE 扫描
        Main->>GF: connect(addr)
        GF->>BLE: Peripheral.connect(addr)，setMTU(2000)
        GF->>BLE: getCharacteristic(CMD_UUID / NOTIFY_UUID)
        GF->>BLE: setDelegate(MyDelegate)
    end
    Main->>Ros: rospy.init_node(gf_pub_node)，创建 5 个 Publisher
    Main->>GF: setDataNotifSwitch(ACCELERATE | EMG_RAW | QUATERNION, set_cmd_cb, 2000)
    GF->>Dev: sendCommand 写 CMD_SET_DATA_NOTIF_SWITCH
    Dev-->>GF: 命令响应（_onResponse 后调用 set_cmd_cb）
    Main->>GF: startDataNotification(onData)
    GF->>BLE: setNotify(notifyCharacteristic, True)
    Main->>Main: rospy.spin()
    loop 每个 BLE 通知
        Dev-->>BLE: notification
        BLE->>GF: MyDelegate.handleNotification，再调 handleDataNotification
        GF->>Main: onData(fullPacket)
        alt NTF_EMG_ADC_DATA
            Main->>Ros: emgPub.publish(EmgArray) 每包 16 条
        else NTF_QUAT_FLOAT_DATA
            Main->>Ros: quatPub.publish(Quaternion)
        else NTF_ACC_DATA
            Main->>Ros: accPub.publish(ImuArray)
        else NTF_GYO_DATA
            Main->>Ros: gyroPub.publish(ImuArray)
        else NTF_EULER_DATA
            Main->>Ros: eulerPub.publish(Euler)
        end
    end
    Note over Main: 开关未含 DNF_GYROSCOPE / DNF_EULERANGLE，gyro / euler 分支当前不会触发
    Note over Main: 退出时 stopDataNotification，setDataNotifSwitch(DNF_OFF)，disconnect
```

图依据：ros_gforce@1b91a0f，`src/gforce_pub.py`、`src/gforce.py`（connect / scan / setDataNotifSwitch / startDataNotification / handleDataNotification 段）、`CMakeLists.txt`、`README.md`；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [ros_gforce 源码归档](../../sources/repos/ros_gforce.md)（<https://github.com/oymotion/ros_gforce>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：oymotion/ros_gforce@1b91a0f，读过 README.md、CMakeLists.txt、src/gforce_pub.py、src/gforce.py（connect / scan / 通知分发相关段落）、src/sample.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
