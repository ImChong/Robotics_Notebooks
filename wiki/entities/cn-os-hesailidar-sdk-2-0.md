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
  - ../../sources/repos/hesailidar_sdk_2_0.md
summary: "禾赛科技 开源项目 HesaiLidar_SDK_2.0（SDK/驱动）：提供禾赛激光雷达数据接收、解析和点云输出SDK，为机器人定位、感知和避障提供传感器输入。…"
institutions:
  - hesai
---

# HesaiLidar_SDK_2.0

## 一句话定义

**HesaiLidar_SDK_2.0** 是 [禾赛科技](https://github.com/HesaiTechnology) 公开的 **SDK/驱动** 开源项目：提供禾赛激光雷达数据接收、解析和点云输出SDK，为机器人定位、感知和避障提供传感器输入。

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
| 机构 | 禾赛科技 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/HesaiTechnology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `HesaiLidar_SDK_2.0` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`cd build && cmake .. && make`，运行示例 `./sample ../config/sample_config.example.ini`（可加第二个参数 `1` 启用 GPU）；源码为 `test/test.cc`。数据源由 INI 里的 `source_type` 决定（实时 UDP、PCAP、串口、TCP、ROS 包），下图画实时 UDP 路径。

```mermaid
sequenceDiagram
    participant T as test/test.cc main()
    participant SDK as driver/hesai_lidar_sdk.hpp HesaiLidarSdk
    participant LD as libhesai/Lidar/lidar.h Lidar
    participant SRC as libhesai/Source SocketSource
    participant PTC as libhesai/PtcClient PtcClient
    participant UP as libhesai/UdpParser（udpX_Y_parser）
    T->>T: LoadIniMap、ApplyToDriverParam 填 DriverParam
    T->>SDK: RegRecvCallback(lidarCallback、faultMessageCallback)
    T->>SDK: Init(param)
    SDK->>LD: 起 InitThread：lidar_ptr_->Init(param)
    LD->>SRC: new SocketSource(udp_port, host_ip, multicast) 并 Open
    LD->>LD: SetThreadNum 起 ReceiveUdpThread 与 ParserThread
    LD->>PTC: new PtcClient（use_ptc_connected 时）并起 InitSetPtc 线程
    LD->>LD: 读到首个包后 DecodePacket 识别雷达型号，创建对应 UdpParser
    LD->>PTC: LoadCorrectionForUdpParser 从雷达取校正文件；失败则 LoadCorrectionFile(本地 correction_file_path)
    T->>SDK: Start()：待初始化完成后起 Run 线程
    loop ReceiveUdpThread
        SRC-->>LD: source_->Receive 得到 UDP 包，放入 origin_packets_buffer_
    end
    loop Run 线程
        SDK->>LD: GetOnePacket
        SDK->>LD: DecodePacket（解析距离、方位、反射率等）
        LD->>UP: ComputeXYZI（ParserThread 或 HandleThread 并行算 x y z intensity；use_cuda 时走 gpu_parser_ptr_）
        SDK->>SDK: scan_complete 后 FrameProcess，MultiSend
        SDK-->>T: point_cloud_cb_(frame) 即 lidarCallback
    end
    T->>T: 等待 IsPlayEnded 或丢包统计时长到，再 sample.Stop()
```

图依据：HesaiTechnology/HesaiLidar_SDK_2.0@fbcd94a，读过 README.md、test/test.cc、driver/hesai_lidar_sdk.hpp、libhesai/Lidar/lidar.h（Init、ReceiveUdpThread、ParserThread、HandleThread 段）、CMakeLists.txt（可执行目标）。各型号 `UdpParser/*` 内部解码公式、`tool/`、`tool_ptc/`、GPU `.cu` 实现未读，图中未画。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [HesaiLidar_SDK_2.0 源码归档](../../sources/repos/hesailidar_sdk_2_0.md)（<https://github.com/HesaiTechnology/HesaiLidar_SDK_2.0>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HesaiTechnology/HesaiLidar_SDK_2.0@fbcd94a，读过 README.md、test/test.cc、driver/hesai_lidar_sdk.hpp、libhesai/Lidar/lidar.h、CMakeLists.txt（可执行目标部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [禾赛科技 官方组织](https://github.com/HesaiTechnology)
