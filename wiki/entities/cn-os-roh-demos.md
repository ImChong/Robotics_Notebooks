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
  - ../../sources/repos/roh_demos.md
summary: "傲意科技 开源项目 roh_demos（工程与工具）：ROHand 灵巧手演示工程…"
institutions:
  - ohand
---

# roh_demos

## 一句话定义

**roh_demos** 是 [傲意科技](https://github.com/oymotion) 公开的 **工程与工具** 开源项目：ROHand 灵巧手演示工程

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
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/oymotion |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `roh_demos` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：在 `gesture_ctrled_rohand/` 目录下运行 `python gesture_ctrled_hand.py`（`NODE_ID = 2`，按 `q` 退出）。本图只取手势控制演示；仓库另有 glove、gForce、loop_test 三个演示，未画入图。

```mermaid
sequenceDiagram
    participant M as gesture_ctrled_hand.py main 主线程
    participant CT as camera_thread 守护线程
    participant CAM as cv2.VideoCapture 摄像头
    participant HD as HandTrackingModule HandDetector
    participant Q as gesture_queue 与 image_queue
    participant MB as pymodbus ModbusSerialClient
    participant DEV as ROHand 灵巧手 Modbus 从站

    M->>M: find_comport 按 CH340 或 USB 查找串口
    M->>MB: ModbusSerialClient 串口 RTU 115200 connect
    M->>CT: threading.Thread 启动 camera_thread

    loop camera_thread while True
        CT->>CAM: video.read()
        CAM-->>CT: 帧 img 经 cv2.flip 镜像
        CT->>HD: findHands(img, draw=True)
        HD-->>CT: 手部关键点 lmlist
        CT->>HD: fingersUp(lmlist 首只手)
        HD-->>CT: 五指伸展状态列表
        CT->>CT: 按手势选取 gestures 图标并生成六路目标 gesture
        CT->>CT: 拇指防抖 interval 为 10 帧
        CT->>Q: put gesture 与 put img 队列已满则跳过
    end

    loop 主线程 while True
        M->>Q: gesture_queue.get() 阻塞取手势
        M->>Q: image_queue 非空时取图并 cv2.imshow Video
        alt 手势与上一帧不同
            M->>M: 0.7 秒内变化则拇指置 0
            M->>MB: write_registers(client, ROH_FINGER_POS_TARGET0, gesture)
            MB->>DEV: Modbus RTU 写保持寄存器 1135 站号 2
            DEV-->>MB: 响应
            MB-->>M: resp.isError 为假则成功
        end
        M->>M: cv2.waitKey(1) 检测到 q 则跳出
    end
    M->>MB: client.close()
```

图依据：roh_demos@ca7d67f，读过 README.md、gesture_ctrled_rohand/README.md、gesture_ctrled_rohand/gesture_ctrled_hand.py（全文）、gesture_ctrled_rohand/HandTrackingModule.py（类与 fingersUp 签名）、common/roh_registers_v1.py（`ROH_FINGER_POS_TARGET0` 定义行）、gesture_ctrled_rohand/requirements.txt。未运行代码（需摄像头与真机）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [roh_demos 源码归档](../../sources/repos/roh_demos.md)（<https://github.com/oymotion/roh_demos>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：oymotion/roh_demos@ca7d67f，读过 README.md、gesture_ctrled_rohand/README.md、gesture_ctrled_hand.py、HandTrackingModule.py、common/roh_registers_v1.py、gesture_ctrled_rohand/requirements.txt；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
