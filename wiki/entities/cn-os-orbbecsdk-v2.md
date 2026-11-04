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
  - ../../sources/repos/orbbecsdk_v2.md
summary: "奥比中光 开源项目 OrbbecSDK_v2（SDK/驱动）：提供奥比中光RGB-D相机底层SDK，为机器人感知、标定和数据采集读取图像与深度流。…"
institutions:
  - orbbec
---

# OrbbecSDK_v2

## 一句话定义

**OrbbecSDK_v2** 是 [奥比中光](https://github.com/orbbec) 公开的 **SDK/驱动** 开源项目：提供奥比中光RGB-D相机底层SDK，为机器人感知、标定和数据采集读取图像与深度流。

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
| 机构 | 奥比中光 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/orbbec |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `OrbbecSDK_v2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 源码运行时序图

入口：编译 `examples/0.basic.quick_start` 后运行其生成的 `quick_start` 可执行文件。本图只覆盖仓库内示例程序对 SDK 的调用；`libOrbbecSDK` 核心实现在仓库中是预编译库（仓库只含头文件与 `.so`），其内部流程不在本图范围内，不臆造。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant QS as quick_start.cpp（main）
    participant P as ob::Pipeline（libobsensor 头文件 / 预编译 SDK）
    participant W as ob_smpl::CVWindow（examples/utils/utils_opencv.cpp）
    U->>QS: 运行 quick_start
    QS->>P: ob::Pipeline pipe
    QS->>P: pipe.start()（默认配置，可由 *SDKConfig.xml 修改）
    Note over P: 预编译 libOrbbecSDK 内部，仓库内无核心源码
    QS->>W: CVWindow win("QuickStart", 1280, 720, ARRANGE_ONE_ROW)
    loop while win.run() 为真
        W->>W: run()：cv::imshow，cv::waitKey(1)
        QS->>P: pipe.waitForFrameset()（默认超时 1000ms）
        P-->>QS: frameSet
        QS->>W: win.pushFramesToView(frameSet)
    end
    QS->>P: pipe.stop()
```

图依据：orbbec/OrbbecSDK_v2@f5d5993，读过 `README.md`、`examples/0.basic.quick_start/quick_start.cpp`、`examples/0.basic.quick_start/CMakeLists.txt`、`examples/utils/utils_opencv.cpp`（部分，`CVWindow::run` 一带）；未运行代码。

## 参考来源

- [OrbbecSDK_v2 源码归档](../../sources/repos/orbbecsdk_v2.md)（<https://github.com/orbbec/OrbbecSDK_v2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：orbbec/OrbbecSDK_v2@f5d5993，读过 `README.md`、`examples/0.basic.quick_start/quick_start.cpp`、`examples/0.basic.quick_start/CMakeLists.txt`、`examples/utils/utils_opencv.cpp`（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [奥比中光 官方组织](https://github.com/orbbec)
