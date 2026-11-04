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
  - ../../sources/repos/wuji-retargeting.md
summary: "舞肌科技 开源项目 wuji-retargeting（动作重定向）：把Vision Pro获得的人手追踪结果映射为Wuji Hand可执行关节目标，为遥操作和示范采集提供人手到机器人手的映射入口。…"
institutions:
  - wuji-robotics
---

# wuji-retargeting

## 一句话定义

**wuji-retargeting** 是 [舞肌科技](https://github.com/wuji-technology) 公开的 **动作重定向** 开源项目：把Vision Pro获得的人手追踪结果映射为Wuji Hand可执行关节目标，为遥操作和示范采集提供人手到机器人手的映射入口。

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
| 类别 | 动作重定向 |
| 官方组织 | https://github.com/wuji-technology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wuji-retargeting` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`cd example && python teleop_real.py --input wuji_glove --hand right --glove-sn <YOUR_SN>`（真机；仿真用 `teleop_sim.py`）。

```mermaid
sequenceDiagram
    autonumber
    participant T as example/teleop_real.py main 与 run_teleop
    participant G as input_devices/wuji_glove_device.py WujiGloveDevice
    participant R as wuji_retargeting/retarget.py Retargeter
    participant O as wuji_retargeting/opt/adaptive_analytical.py AdaptiveOptimizerAnalytical
    participant F as wuji_retargeting/opt/base.py LPFilter
    participant B as teleop_real.py WujiHandBackend 或 WujiHand2Backend

    T->>G: create_wuji_glove_device 构造 WujiGloveDevice(hand_side, device_name, sn)
    T->>R: Retargeter.from_yaml(config, hand_side)
    R->>O: BaseOptimizer.from_config 构造优化器
    O->>O: RobotWrapper(urdf_path) 用 pinocchio 加载手部模型
    R->>F: LPFilter(lp_alpha)
    T->>B: 最后才上电: WujiHandBackend 或 WujiHand2Backend
    loop 直到 Ctrl+C
        T->>G: get_fingers_data()
        G-->>T: 最新 (21, 3) 关键点, 取 left 或 right 侧
        alt 首帧为空或全零
            T->>T: sleep 10 ms 后跳过
        else 有效帧
            T->>R: retarget(fingers_pose)
            R->>R: apply_mediapipe_transformations 坐标变换到手腕系
            R->>R: _apply_rotation 与 _apply_offset (仅配置非零时)
            R->>O: solve(mediapipe_kp)
            O->>O: _compute_pinch_alpha 与目标向量计算
            O->>O: _run_optimization NLopt 求解 qpos
            O-->>R: qpos (20,)
            R->>F: next(qpos) 低通滤波
            F-->>R: 滤波后 qpos
            R-->>T: qpos
            T->>T: 若有 mjcf_path 则按关节名重排 qpos
            T->>B: send(qpos)
            B->>B: 下发关节目标位置到真机
        end
    end
    T->>B: close() 失能关节或断开会话
    T->>G: cleanup() 或 close() 清理输入设备
```

图依据：wuji-technology/wuji-retargeting@531f6ed，实际读过 `README.md`、`example/teleop_real.py`、`example/input_devices/wuji_glove_device.py`（`get_fingers_data`）、`wuji_retargeting/retarget.py`、`wuji_retargeting/robot.py`、`wuji_retargeting/opt/base.py`（`from_config`、`_run_optimization`）、`wuji_retargeting/opt/adaptive_analytical.py`（`solve`）。`mediapipe.py`、`LPFilter` 等未逐行读，仅据调用名画图。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wuji-retargeting 源码归档](../../sources/repos/wuji-retargeting.md)（<https://github.com/wuji-technology/wuji-retargeting>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：wuji-technology/wuji-retargeting@531f6ed，读过 README.md、example/teleop_real.py、example/input_devices/wuji_glove_device.py、wuji_retargeting/retarget.py、wuji_retargeting/robot.py、wuji_retargeting/opt/base.py、wuji_retargeting/opt/adaptive_analytical.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [舞肌科技 官方组织](https://github.com/wuji-technology)
