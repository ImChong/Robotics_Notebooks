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
  - ../../sources/repos/xensesdk.md
summary: "千觉机器人 开源项目 xensesdk（SDK/驱动）：Xense 触觉传感器 Python SDK：深度、形变、标记点跟踪与力场数据…"
institutions:
  - qianjue
---

# xensesdk

## 一句话定义

**xensesdk** 是 [千觉机器人](https://github.com/XenseRobotics-AI) 公开的 **SDK/驱动** 开源项目：Xense 触觉传感器 Python SDK：深度、形变、标记点跟踪与力场数据

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
| 机构 | 千觉机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/XenseRobotics-AI、https://github.com/XenseRobotics |

### 架构流程图

```mermaid
flowchart LR
  SRC["数据源: USB 传感器 (SN/相机ID) / 远程 IP / 离线视频 data.h5"] --> INIT["Sensor.create / scanSerialNumber / calibrateSensor"]
  INIT --> INFER["onnxruntime_gpu 推理后端 (CUDA/cuDNN)"]
  INFER --> SEL["selectSensorInfo (同帧同步枚举读取)"]
  SEL --> OUT1["图像与深度: Rectify / Difference / Depth"]
  SEL --> OUT2["形变与标记: Marker2D / Mesh3D / Mesh3DInit / Mesh3DFlow"]
  SEL --> OUT3["三维力场: Force / ForceNorm / ForceResultant (6D)"]
  INIT --> CFG["exportRuntimeConfig -> createSolver 离线重放"]
  SEL --> REL["release() 释放资源 / call_service 算力板服务"]
```

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `xensesdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 源码运行时序图

以下时序图基于对官方示例脚本 `Examples/example_local.py`、`Examples/example_remote.py` 与 `Examples/example_solver.py` 的实际调用链梳理：

```mermaid
sequenceDiagram
  participant App as example_local / example_remote / example_solver
  participant RPC as call_service (xensesdk)
  participant S as Sensor (xensesdk)
  participant Sol as Solver (createSolver)
  participant UI as ExampleView (view2d / Plot)

  alt 远程连接 (Examples/example_remote.py)
    App->>RPC: call_service("master_...", "scan_sensor_sn", timeout_sec=1)
    RPC-->>App: 返回设备序列号字典 (serial_number)
    App->>S: Sensor.create(serial_number, mac_addr=...)
  else 本地连接 (Examples/example_local.py)
    App->>S: Sensor.create(cam_id=0)
  else Solver 分离推理 (Examples/example_solver.py)
    App->>S: Sensor.create("OG000237")
    App->>S: sensor.exportRuntimeConfig(save_dir=".")
    App->>Sol: Sensor.createSolver("./runtime_OG000237")
    App->>S: sensor.selectSensorInfo(OutputType.Rectify)
    App->>Sol: solver.calibrateSensor(rectify_image=rectify)
  end

  App->>UI: ExampleView(..., title=...) / create2d(...) / createPlot("Force", "Torque")
  App->>UI: view.setCallback(update) -> view.show()

  loop 逐帧回调 update()
    alt 本地或远程直连
      UI->>S: sensor.selectSensorInfo(Rectify, Difference, Depth, Force, ForceResultant)
      S-->>UI: rectify, difference, depth, force, force6d
      UI->>S: sensor.drawMarkerMove(rectify)
    else Solver 推理路径
      UI->>S: sensor.selectSensorInfo(OutputType.Rectify)
      UI->>Sol: solver.selectSensorInfo(..., rectify_image=rectify)
      Sol-->>UI: rectify, difference, depth, force, force_resultant
      UI->>Sol: solver.drawMarkerMove(rectify)
    end
    UI->>UI: view2d.setData / setDepth / setForceFlow / force_plot / torque_plot
  end

  App->>S: finally sensor.release()
```

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [xensesdk 源码归档](../../sources/repos/xensesdk.md)（<https://github.com/XenseRobotics/xensesdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [千觉机器人 官方组织](https://github.com/XenseRobotics-AI)
