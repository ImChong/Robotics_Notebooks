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
  - ../../sources/repos/fourier_dexhand_sdk.md
summary: "傅利叶智能 开源项目 fourier_dexhand_sdk（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - fourier
---

# fourier_dexhand_sdk

## 一句话定义

**fourier_dexhand_sdk** 是 [傅利叶智能](https://github.com/FFTAI) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 傅利叶智能 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/FFTAI |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `fourier_dexhand_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：C++ 示例 `example/cpp/build` 后运行 `../bin/fourier_dexhand_sdk_demo`；Python 示例 `python3 example/python/fourier_dexhand_sdk_demo.py`（依赖 pip 包 `dexhandpy`）。本仓库只含头文件、预编译 `libFourierDexHand.so` 和示例，`DexHand`/`Fhand` 的实现体不在仓库内，下图只画头文件声明与示例 `main()` 能确认的调用。

```mermaid
sequenceDiagram
    participant M as fourier_dexhand_sdk_demo.cpp main()
    participant D as FdHand::DexHand (dexhand.h)
    participant F as hand_ws::Fhand (hand/fhand.h)
    participant B as BaseHand 子类 fdhv1/fdhv2/inspire (fourierdexhand/)
    participant H as 灵巧手设备（UDP，默认 192.168.137.19/39）
    M->>D: 构造 DexHand 后 init()
    D->>F: Fhand::init() 搜索设备（broadcast/find_hand 为私有声明）
    F-->>D: ip_unique/name/type/driver_ver/hardware_ver
    D-->>M: Ret::SUCCESS
    M->>D: get_ip_list()
    D-->>M: 设备 IP 列表
    loop 每个 IP 一个 std::thread
        M->>D: get_name/get_type/get_driver_ver/get_hardware_ver(ip)
        M->>D: get_pos(ip)
        D->>F: get_position(ip, position)
        F->>B: get_pos(fdb)
        B->>H: UDP 请求，端口 2333 控制/2334 通信/2335 fast（basehand.h 常量）
        H-->>B: 响应，由 BaseHandProtocol::HandProtocol 解析并校验 CRC
        M->>D: get_pvc(ip) / get_hand_config(ip)
        M->>D: reboot(ip)
        D->>F: reboot(ip)
    end
    Note over D,B: 具体 UDP 收发与 BaseHand 子类选择逻辑在 libFourierDexHand.so 内，源码未开放，不画
```

图依据：FFTAI/fourier_dexhand_sdk@e8e831c，读过 README.md、example/cpp/fourier_dexhand_sdk_demo.cpp、example/cpp/CMakeLists.txt、example/cpp/include/dexhand.h、hand/fhand.h、hand/fourierdexhand/basehand.h、handprotocol.h、example/python/fourier_dexhand_sdk_demo.py。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [fourier_dexhand_sdk 源码归档](../../sources/repos/fourier_dexhand_sdk.md)（<https://github.com/FFTAI/fourier_dexhand_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：FFTAI/fourier_dexhand_sdk@e8e831c，读过 README.md、example/cpp/fourier_dexhand_sdk_demo.cpp、example/cpp/CMakeLists.txt、example/cpp/include/dexhand.h、hand/fhand.h、hand/fourierdexhand/basehand.h、handprotocol.h、example/python/fourier_dexhand_sdk_demo.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傅利叶智能 官方组织](https://github.com/FFTAI)
