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
  - ../../sources/repos/data_tools.md
summary: "松灵机器人 开源项目 data_tools（数据集/Benchmark）：数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈…"
institutions:
  - agilex
---

# data_tools

## 一句话定义

**data_tools** 是 [松灵机器人](https://github.com/agilexrobotics) 公开的 **数据集/Benchmark** 开源项目：数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

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
| 机构 | 松灵机器人 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/agilexrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `data_tools` 仓库并核对 README 许可与依赖。
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

入口：`roslaunch data_tools run_data_capture.launch type:=aloha datasetDir:=<路径> episodeIndex:=0`（采集，ROS1 noetic），之后离线执行 `python3 data_sync.py --type aloha --datasetDir <路径>` 与 `python3 data_to_hdf5.py --type aloha --datasetDir <路径>`（命令出自 README）

```mermaid
sequenceDiagram
    participant L as run_data_capture.launch
    participant C as src/dataCapture.cpp main / DataCapture
    participant T as ROS 话题（相机、关节、末端位姿、夹爪等）
    participant FS as datasetDir/episodeN 磁盘目录
    participant S as scripts/data_sync.py Operator
    participant H as scripts/data_to_hdf5.py

    L->>L: rosparam 加载 config/type_data_params.yaml，设置 datasetDir / episodeIndex / hz / timeout / cropTime
    L->>C: 启动 data_tools_dataCapture 节点
    C->>C: nh.param 读参数，构造 DataCapture(datasetDir, episodeIndex, hz, timeout)
    C->>FS: instructionSaving(instructions)
    C->>T: 按 yaml 配置逐个 nh.subscribe（图像、点云、joint_state、end_pose、gripper、imu 等）
    C->>C: run()：每路话题一个 *Saving 线程，另有 keyboardInterruptChecking 与 monitoring 线程
    loop 话题回调（cameraColorHandler 等）
        T-->>C: 消息进入对应 MsgDeque，更新计数与时间戳
        C->>FS: *Saving 线程弹出消息，cv::imwrite 等写文件，CameraInfo 写 config.json
    end
    C->>C: 用户按 ENTER 后 keyboardInterruptChecking 调 shutdown：置 captureStop，退订并压入时间戳为 0 的哨兵消息
    C->>FS: 各 Saving 线程排空队列后退出，join() 返回，打印 Done
    Note over S,H: 以下为离线处理，各自独立进程
    S->>FS: 读取 episode 内各类文件，按时间戳对齐 sync()，write_sync_files()
    H->>FS: 读取对齐后的数据，h5py.File(dataFile, 'w') 写 hdf5
```

图依据：agilexrobotics/data_tools@58d0ec7，读过 `README.md`、`launch/run_data_capture.launch`、`launch/run_data_sync.launch`、`launch/run_data_publish.launch`、`src/dataCapture.cpp`（`main`、`run`、`cameraColorHandler`、`cameraColorSaving`、`keyboardInterruptChecking`/`shutdown`、订阅段）、`scripts/data_sync.py` 与 `scripts/data_to_hdf5.py`（仅看类/函数签名与参数，内部细节未读）。`dataSync.cpp`、`dataPublish.cpp`、`data_publish.py`、`hdf5_to_lerobot.py` 等未读，也未画入。

## 参考来源

- [data_tools 源码归档](../../sources/repos/data_tools.md)（<https://github.com/agilexrobotics/data_tools>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：agilexrobotics/data_tools@58d0ec7，读过 README.md、launch/run_data_capture.launch、launch/run_data_sync.launch、launch/run_data_publish.launch、src/dataCapture.cpp、scripts/data_sync.py（部分）、scripts/data_to_hdf5.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [松灵机器人 官方组织](https://github.com/agilexrobotics)
