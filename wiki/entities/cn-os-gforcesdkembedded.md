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
  - ../../sources/repos/gforcesdkembedded.md
summary: "傲意科技 开源项目 gForceSDKEmbedded（SDK/驱动）：gForce 肌电臂环 C/C++ 嵌入式 SDK…"
institutions:
  - ohand
---

# gForceSDKEmbedded

## 一句话定义

**gForceSDKEmbedded** 是 [傲意科技](https://github.com/oymotion) 公开的 **SDK/驱动** 开源项目：gForce 肌电臂环 C/C++ 嵌入式 SDK

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

1. 从官方 GitHub/Gitee 组织检索 `gForceSDKEmbedded` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：仓库不含可执行 `main()`。应用程序（嵌入式 MCU 的 `main()`）按 README 示例调用 `GF_Init(SYS_GetChar, SYS_GetTick)` 后循环调用 `GF_GetGForceData()`；`SYS_GetChar` / `SYS_GetTick` 由用户按平台自行实现。`gForceAdapterC.c` 为可直接编入工程的 C 适配层。

```mermaid
sequenceDiagram
    participant APP as 应用 main()（README 示例，用户编写）
    participant AD as gForceAdapterC.c（GF_* 解析层）
    participant SYS as SYS_GetChar / SYS_GetTick（用户平台实现）
    participant UART as UART 串口（gForceJoint 输出）
    APP->>AD: GF_Init(SYS_GetChar, SYS_GetTick)（保存两个回调指针）
    loop while(true)，每次 timeout = 100ms
        APP->>AD: GF_GetGForceData(&gForceData, 100)
        loop 逐字节读取，直到帧头与帧完整
            AD->>SYS: m_getCharFunc(&byte)
            SYS->>UART: 读取 RX 数据（0 表示无数据）
            UART-->>SYS: 1 字节
            SYS-->>AD: 返回 1（有数据）或 0
            AD->>SYS: m_getTimerFunc() 判断是否超时
        end
        alt 类型或长度非法
            AD-->>APP: GF_RET_ERR_DATA
        else 超时，或同一手势重复（静默丢弃帧头不匹配的字节）
            AD-->>APP: GF_RET_TIME_OUT
        else 解析完成
            AD-->>APP: GF_RET_OK，type = GF_QUATERNION / GF_GESTURE
            alt type == GF_QUATERNION（16 字节）
                APP->>AD: GF_QuaternionToEuler(&quat, &euler)
                AD-->>APP: euler（欧拉角）
            else type == GF_GESTURE
                APP->>APP: switch(gesture) 处理手势
            end
        end
    end
```

注：`GF_GotGesture(gesture, timeout)` 是对 `GF_GetGForceData` 的手势判定封装（README 所述 C++ 等价用法见 `gForceAdaptor`，本仓库未包含）；`GF_EMGRAW` 类型在头文件中定义，但 README 示例循环未处理。

图依据：gForceSDKEmbedded@74aeeda，读过 README.md、gForceAdapterC.c（GF_Init、GF_GotGesture、GF_GetGForceData 的解析与超时逻辑）、gForceDataTypes.h（事件类型常量，grep 读取）；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [gForceSDKEmbedded 源码归档](../../sources/repos/gforcesdkembedded.md)（<https://github.com/oymotion/gForceSDKEmbedded>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：oymotion/gForceSDKEmbedded@74aeeda，读过 README.md、gForceAdapterC.c、gForceDataTypes.h（grep）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
