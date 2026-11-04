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
  - ../../sources/repos/emgfilters.md
summary: "傲意科技 开源项目 EMGFilters（工程与工具）：肌电信号滤波算法库…"
institutions:
  - ohand
---

# EMGFilters

## 一句话定义

**EMGFilters** 是 [傲意科技](https://github.com/oymotion) 公开的 **工程与工具** 开源项目：肌电信号滤波算法库

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

1. 从官方 GitHub/Gitee 组织检索 `EMGFilters` 仓库并核对 README 许可与依赖。
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

入口：Arduino 草图 `examples/SimpleEMGFilters/SimpleEMGFilters.ino`（默认 `#define CALIBRATE 1`，即先打印平方值用于标定基线；改为 0 后才按 `baseline` 门限输出包络）。

```mermaid
sequenceDiagram
    autonumber
    participant MCU as SimpleEMGFilters.ino setup/loop
    participant ADC as analogRead A0
    participant EMG as EMGFilters::update
    participant AHF as AHF FILTER_4th 陷波 50Hz 或 60Hz
    participant LPF as LPF FILTER_2nd 低通 150Hz
    participant HPF as HPF FILTER_2nd 高通 20Hz
    participant SER as Serial

    MCU->>EMG: init(SAMPLE_FREQ_1000HZ, NOTCH_FREQ_50HZ, true, true, true)
    EMG->>AHF: init 选择 500 或 1000Hz 系数
    EMG->>LPF: init(FILTER_TYPE_LOWPASS, 采样率)
    EMG->>HPF: init(FILTER_TYPE_HIGHPASS, 采样率)
    MCU->>SER: Serial.begin(115200)

    loop 每个采样周期 interval
        MCU->>ADC: analogRead(SensorInputPin)
        ADC-->>MCU: data
        MCU->>EMG: update(data)
        alt 采样率或工频不受支持 m_bypassEnabled
            EMG-->>MCU: 原样返回输入
        else 正常滤波
            EMG->>AHF: update 陷波 若启用
            AHF-->>EMG: 陷波输出
            EMG->>LPF: update 低通 若启用
            LPF-->>EMG: 低通输出
            EMG->>HPF: update 高通 若启用
            HPF-->>EMG: 滤波输出
        end
        EMG-->>MCU: dataAfterFilter
        MCU->>MCU: envelope = sq(dataAfterFilter)
        alt CALIBRATE 为 1
            MCU->>SER: 打印 Squared Data 用于标定
        else 正常运行
            MCU->>MCU: envelope 低于 baseline 则置零
            MCU->>SER: 打印 envelope
        end
        MCU->>MCU: delay 剩余的 interval 时间
    end
```

图依据：oymotion/EMGFilters@2808974，读过 `README.md`、`EMGFilters.h`（全文）、`EMGFilters.cpp`（滤波器系数表、`FILTER_2nd` / `FILTER_4th` 的 `init` 与 `update`、`EMGFilters::init` 与 `EMGFilters::update` 段）、`examples/SimpleEMGFilters/SimpleEMGFilters.ino`（`setup` 与 `loop` 段）。未编译或烧录，未运行代码。时序图依据以上文件。

## 参考来源

- [EMGFilters 源码归档](../../sources/repos/emgfilters.md)（<https://github.com/oymotion/EMGFilters>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：oymotion/EMGFilters@2808974，读过 `README.md`、`EMGFilters.h`、`EMGFilters.cpp`（滤波器与 init / update 段）、`examples/SimpleEMGFilters/SimpleEMGFilters.ino`（setup / loop 段）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
