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
  - ../../sources/repos/rs_driver.md
summary: "速腾聚创 开源项目 rs_driver（SDK/驱动）：提供跨平台雷达驱动内核，为上层ROS或自定义应用解析速腾聚创雷达数据。…"
institutions:
  - robosense
---

# rs_driver

## 一句话定义

**rs_driver** 是 [速腾聚创](https://github.com/RoboSense-LiDAR) 公开的 **SDK/驱动** 开源项目：提供跨平台雷达驱动内核，为上层ROS或自定义应用解析速腾聚创雷达数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 速腾聚创 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/RoboSense-LiDAR |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `rs_driver` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：以 `-DCOMPILE_DEMOS=ON` 构建后运行 `demo_online`（源码 `demo/demo_online.cpp`；离线回放为 `demo_pcap`，多雷达为 `demo_online_multi_lidars`）。rs_driver 是 header-only 库，`LidarDriver<T>` 的实现在 `lidar_driver_impl.hpp`。下图为 Linux 在线雷达路径（默认 `select()` 接收）。

```mermaid
sequenceDiagram
    participant M as demo_online.cpp main()
    participant D as LidarDriver / LidarDriverImpl
    participant DF as DecoderFactory + Decoder
    participant IF as InputFactory + InputSock
    participant RT as InputSock::recvPacket 线程
    participant HT as LidarDriverImpl::processPacket 线程
    participant UT as 用户线程 processCloud
    M->>M: 填 RSDriverParam(input_type=ONLINE_LIDAR, msop 6699, difop 7788, imu 6688, lidar_type)
    M->>D: regPointCloudCallback / regExceptionCallback / regImuDataCallback
    M->>D: init(param)
    D->>DF: createDecoder(lidar_type)，注册异常回调与 splitFrame 回调
    D->>IF: createInput(...)，注册 packetGet / packetPut
    D->>IF: input_ptr_->init() 创建 msop/difop/imu socket
    M->>UT: std::thread(processCloud)
    M->>D: start()
    D->>HT: 启动 processPacket 线程
    D->>RT: input_ptr_->start() 启动 recvPacket 线程
    loop 收包
        RT->>D: cb_get_pkt_ 取空 Buffer(packetGet)
        RT->>RT: select() 收 UDP 包
        RT->>D: pushPacket -> packetPut(pkt_queue_)
    end
    loop 解包
        HT->>HT: pkt_queue_.popBatch
        HT->>DF: internalProcessPacket: 按包头分 MSOP / DIFOP / IMU
        DF->>D: MSOP 凑满一帧时调 splitFrame
        D->>M: cb_put_cloud_(driverReturnPointCloudToCallerCallback)
        M->>UT: stuffed_cloud_queue.push，processCloud 取出处理后归还 free_cloud_queue
    end
```

图依据：RoboSense-LiDAR/rs_driver@897b14d，读过 CMakeLists.txt（选项）、demo/CMakeLists.txt、demo/demo_online.cpp、src/rs_driver/driver/lidar_driver_impl.hpp（init、start、packetGet/Put、internalProcessPacket、processPacket、splitFrame）、driver/input/input.hpp 与 unix/input_sock_select.hpp（线程与回调部分）；各型号 decoder_*.hpp 的解码细节、PCAP 输入与 epoll 实现未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [rs_driver 源码归档](../../sources/repos/rs_driver.md)（<https://github.com/RoboSense-LiDAR/rs_driver>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：RoboSense-LiDAR/rs_driver@897b14d，读过 CMakeLists.txt、demo/demo_online.cpp、lidar_driver_impl.hpp、input/input.hpp、input/unix/input_sock_select.hpp；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [速腾聚创 官方组织](https://github.com/RoboSense-LiDAR)
