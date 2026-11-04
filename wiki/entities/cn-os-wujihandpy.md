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
  - ../../sources/repos/wujihandpy.md
summary: "舞肌科技 开源项目 wujihandpy（工程与工具）：以C++核心和Python绑定提供Wuji Hand设备发现、状态读取和控制接口，是连接上层算法与真实灵巧手的基础SDK。…"
institutions:
  - wuji-robotics
---

# wujihandpy

## 一句话定义

**wujihandpy** 是 [舞肌科技](https://github.com/wuji-technology) 公开的 **工程与工具** 开源项目：以C++核心和Python绑定提供Wuji Hand设备发现、状态读取和控制接口，是连接上层算法与真实灵巧手的基础SDK。

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
| 机构 | 舞肌科技 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/wuji-technology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wujihandpy` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python example/joint/3.realtime.py`（先 `pip install wujihandpy`；Linux 需配置 USB 权限）。Python 接口由 `src/main.cpp` / `src/wrapper.hpp` 绑定到 C++ 核心 `wujihandcpp`。

```mermaid
sequenceDiagram
    autonumber
    participant E as example/joint/3.realtime.py main run
    participant W as src/wrapper.hpp 与 src/controller.hpp Python 绑定
    participant F as src/filter.hpp LowPass
    participant H as wujihandcpp device/hand.hpp Hand
    participant P as wujihandcpp protocol/handler.cpp Handler
    participant T as wujihandcpp transport USB CDC

    E->>W: wujihandpy.Hand()
    W->>H: 构造 Hand 并在 C++ 构造函数中打开 USB CDC
    H->>P: 创建 Handler 并 start_transmit_receive 启动 SDO 线程
    P->>T: 经 transport 收发帧
    E->>W: write_joint_enabled(True)
    W->>H: 写关节使能
    H->>P: SDO 写请求由 sdo_thread 发送
    E->>W: hand.realtime_controller(enable_upstream, LowPass 5 Hz)
    W->>F: LowPass.create_controller
    F->>H: realtime_controller 构造 IController 包装
    alt 固件滤波可用
        H->>H: 写 PositionFilterCutoffFreq 使用兼容控制器
    else 软件滤波
        H->>P: attach_realtime_controller 启动 pdo_thread
        P->>P: pdo_thread_main 以 500 Hz TickExecutor 循环
    end
    loop 每个周期约 100 Hz 的上层循环
        E->>W: controller.set_joint_target_position target 5x4
        W->>H: FilteredControllerOperator.set 写入滤波器输入
        E->>W: get_joint_actual_position 与 get_joint_actual_effort
        W->>P: 读取 realtime 原子缓存 pdo_read_position_ 等
        E->>E: 计算误差与力矩百分比并打印
    end
    loop pdo_thread 500 Hz
        P->>P: FilteredController.step 生成本周期目标
        P->>P: pdo_write_async_unchecked 组装 PDO 写帧
        P->>T: 帧交给 transport 发送
    end
    E->>W: finally write_joint_enabled(False)
    W->>H: 失能全部关节
```

图依据：wuji-technology/wujihandpy@468e169，实际读过 `README.md`、`example/joint/3.realtime.py`、`src/wrapper.hpp`（`realtime_controller`、`get_*` 相关片段）、`src/controller.hpp`（`set_joint_target_position`）、`src/filter.hpp`（`LowPass::create_controller`）、`wujihandcpp/include/wujihandcpp/device/hand.hpp`（`realtime_controller`、`attach_realtime_controller`）、`wujihandcpp/include/wujihandcpp/device/controller.hpp`、`wujihandcpp/src/protocol/handler.cpp`（`pdo_thread_main`、`pdo_write_async_unchecked`、`realtime_set_joint_target_position`）。`transport` 与 `Hand` 构造细节只读了函数名，未逐行核对；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wujihandpy 源码归档](../../sources/repos/wujihandpy.md)（<https://github.com/wuji-technology/wujihandpy>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：wuji-technology/wujihandpy@468e169，读过 README.md、example/joint/3.realtime.py、src/wrapper.hpp、src/controller.hpp、src/filter.hpp、wujihandcpp/include/wujihandcpp/device/{hand,controller}.hpp、wujihandcpp/src/protocol/handler.cpp；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [舞肌科技 官方组织](https://github.com/wuji-technology)
