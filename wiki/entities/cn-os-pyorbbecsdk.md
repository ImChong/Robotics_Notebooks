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
  - ../../sources/repos/pyorbbecsdk.md
summary: "奥比中光 开源项目 pyorbbecsdk（SDK/驱动）：提供OrbbecSDK的Python绑定，便于快速构建深度感知、采集和标定工具。…"
institutions:
  - orbbec
---

# pyorbbecsdk

## 一句话定义

**pyorbbecsdk** 是 [奥比中光](https://github.com/orbbec) 公开的 **SDK/驱动** 开源项目：提供OrbbecSDK的Python绑定，便于快速构建深度感知、采集和标定工具。

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

1. 从官方 GitHub/Gitee 组织检索 `pyorbbecsdk` 仓库并核对 README 许可与依赖。
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

入口：`python examples/quick_start.py`（需先 `pip install pyorbbecsdk2 opencv-python numpy` 并连接 Orbbec 相机）。`pyorbbecsdk` 模块由 `src/pyorbbecsdk/*.cpp`（pybind11 绑定）封装 SDK 的 C++ 接口；SDK 核心实现在 `sdk/` 中为头文件与预编译库，内部流程不在本图范围内。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant QS as examples/quick_start.py（main）
    participant UT as examples/utils.py（frame_to_bgr_image）
    participant PB as pyorbbecsdk 绑定（src/pyorbbecsdk/pipeline.cpp）
    participant SDK as libOrbbecSDK（sdk/ 预编译，核心实现不在仓库内）
    participant CV as OpenCV（cv2）
    U->>QS: python examples/quick_start.py
    QS->>PB: Pipeline()，pipeline.start()（无 Config 时读取 config/OrbbecSDKConfig.xml）
    PB->>SDK: Pipeline::start(config 或 nullptr)
    QS->>CV: namedWindow / resizeWindow("QuickStart Viewer")
    loop while True，直到按 Q 或 ESC
        QS->>PB: pipeline.wait_for_frames(1000)
        PB->>SDK: 取帧集（1000 ms 超时）
        SDK-->>PB: FrameSet
        PB-->>QS: frames
        QS->>QS: frames.get_color_frame()，get_depth_frame()
        QS->>UT: frame_to_bgr_image(color_frame)
        QS->>QS: depth get_data → np.frombuffer(uint16) × get_depth_scale → depth_mm
        QS->>QS: render_depth_3d(depth_mm)：Scharr 梯度 + COLORMAP_JET
        QS->>CV: cv2.imshow(combined)，cv2.waitKey(1)
    end
    QS->>PB: pipeline.stop()
    QS->>CV: destroyAllWindows()
```

图依据：orbbec/pyorbbecsdk@68cdfa1，读过 `README.md`、`examples/quick_start.py`、`examples/utils.py`（部分）、`src/pyorbbecsdk/pipeline.cpp`（部分，start / stop / wait_for_frames 绑定）、`src/pyorbbecsdk/frame.cpp`（部分）；未运行代码。

## 参考来源

- [pyorbbecsdk 源码归档](../../sources/repos/pyorbbecsdk.md)（<https://github.com/orbbec/pyorbbecsdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：orbbec/pyorbbecsdk@68cdfa1，读过 `README.md`、`examples/quick_start.py`、`examples/utils.py`（部分）、`src/pyorbbecsdk/pipeline.cpp`（部分）、`src/pyorbbecsdk/frame.cpp`（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [奥比中光 官方组织](https://github.com/orbbec)
