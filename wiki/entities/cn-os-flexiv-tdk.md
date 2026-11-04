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
  - ../../sources/repos/flexiv_tdk.md
summary: "非夕科技 开源项目 flexiv_tdk（数据集/Benchmark）：提供非夕机器人遥操作开发接口，用于构建主从控制、示范采集和机器人学习数据链路。…"
institutions:
  - flexiv
---

# flexiv_tdk

## 一句话定义

**flexiv_tdk** 是 [非夕科技](https://github.com/flexivrobotics) 公开的 **数据集/Benchmark** 开源项目：提供非夕机器人遥操作开发接口，用于构建主从控制、示范采集和机器人学习数据链路。

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
| 机构 | 非夕科技 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/flexivrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `flexiv_tdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（LAN 双机遥操作，主臂与从臂序列号以 `-l` 与 `-f` 传入，`-D` 启用数字输入线程）：`python3 example_py/transparent_cartesian_teleop_lan.py -l 主臂序列号 -f 从臂序列号 -D`

```mermaid
sequenceDiagram
    autonumber
    participant OP as 操作者 键盘与数字输入
    participant M as main 主线程
    participant CT as console_task 线程
    participant DT as 数字输入线程 read_digital_input_task 仅 -D
    participant TL as flexivtdk TransparentCartesianTeleopLAN 闭源库
    M->>M: parse_args() 解析 leader_sn、follower_sn、-D
    M->>TL: TransparentCartesianTeleopLAN(robot_pairs 列表)
    M->>TL: Init()
    M->>TL: Start()
    M->>CT: threading.Thread(console_task).start()
    opt 指定 -D
        M->>DT: threading.Thread(read_digital_input_task).start()
    end
    loop 直到 _stop_event 置位
        CT->>OP: input() 读取一条命令
        CT->>CT: TeleoperationController.handle_command 查命令表
        alt 轴锁命令 x y z q w e X Y Z Q W E
            CT->>TL: SetAxisLockCmd(index, ARM_1, cmd)
        else 力反馈比例 t T c
            CT->>TL: SetWrenchFeedbackScalingFactor(index, ARM_1, factor)
        else 零空间姿态 i I o O
            CT->>TL: SetLeaderNullSpacePosture 或 SetFollowerNullSpacePosture
        else 最大接触力矩 p
            CT->>TL: SetFollowerMaxContactWrench(index, ARM_1, wrench)
        else 启停 r R b B
            CT->>TL: Engage(index, ARM_1, true 或 false) 或 Stop 或 Start
        end
    end
    loop 每 10 ms, 直到 _stop_event 置位
        DT->>TL: digital_inputs(idx) 返回 leader 与 follower 输入
        DT->>TL: Engage(idx, ARM_1, leader_di 首个端口)
    end
    CT-->>M: 控制台线程退出, 置位 _stop_event
    M->>TL: Stop() (finally 中再次确保停止)
    Note over TL: 主从机器人之间的通信与力反馈在闭源库内部完成, 未展开
```

图依据：flexivrobotics/flexiv_tdk@772543f，README.md、example_py/transparent_cartesian_teleop_lan.py。库内部（lib/ 下预编译静态库）闭源，未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [flexiv_tdk 源码归档](../../sources/repos/flexiv_tdk.md)（<https://github.com/flexivrobotics/flexiv_tdk>）

- 源码核对（2026-11-02）：flexivrobotics/flexiv_tdk@772543f，读过 README.md、example_py/transparent_cartesian_teleop_lan.py；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [非夕科技 官方组织](https://github.com/flexivrobotics)
