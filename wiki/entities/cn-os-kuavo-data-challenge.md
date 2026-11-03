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
  - ../../sources/repos/kuavo_data_challenge.md
summary: "乐聚机器人 开源项目 kuavo_data_challenge（数据集/Benchmark）：数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈…"
institutions:
  - leju
---

# kuavo_data_challenge

## 一句话定义

**kuavo_data_challenge** 是 [乐聚机器人](https://github.com/LejuRobotics) 公开的 **数据集/Benchmark** 开源项目：数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

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
| 机构 | 乐聚机器人 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/LejuRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `kuavo_data_challenge` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（三段流水线，均为仓库内 Python 脚本）：`python kuavo_data/CvtRosbag2Lerobot.py`（rosbag → LeRobot）→ `python kuavo_train/train_policy.py`（训练）→ `python kuavo_deploy/src/scripts/script.py --task go_run --config <yaml>`（ROS 部署推理）。

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant CVT as CvtRosbag2Lerobot.py main
    participant RD as kuavo_dataset.KuavoRosbagReader
    participant LDS as LeRobotDataset
    participant TR as train_policy.py main
    participant POL as CustomDiffusion/ACTPolicyWrapper
    participant SC as script.py ArmMove
    participant ENV as KuavoBaseRosEnv (gym)
    participant EV as real_single_test.kuavo_eval
    participant ROB as Kuavo 机器人 (ROS / KuavoSDK)

    U->>CVT: hydra 配置 KuavoRosbag2Lerobot.yaml
    CVT->>RD: list_bag_files(rosbag_dir)
    CVT->>LDS: create_empty_dataset_chunked(robot_type=kuavo4pro)
    loop populate_dataset_chunked 按 chunk_size 帧分块
        CVT->>RD: 读 rosbag 并对齐帧 (on_frame / on_chunk_done 回调)
        CVT->>LDS: 写入 parquet 与视频
    end
    U->>TR: hydra 配置 diffusion_config / act_config
    TR->>LDS: LeRobotDatasetMetadata 读取 features 与统计量
    TR->>POL: build_policy(policy_name, policy_cfg)
    TR->>TR: make_pre_post_processors 并 save_pretrained
    TR->>POL: 训练循环 (DataLoader 取批, 优化器更新)
    U->>SC: --task go_run --config yaml
    SC->>ROB: rospy.init_node('kuavo_deploy')
    SC->>ENV: gym.make(config.env.env_name)
    SC->>ROB: go(): 插值到 go_bag 首帧并 play_bag 回放
    SC->>EV: run(): kuavo_eval(config, env)
    EV->>POL: setup_policy + build_pre_post_processors
    loop 每个 step 直到 terminated/truncated
        EV->>POL: select_action(observation)
        EV->>ENV: env.step(numpy_action)
        ENV->>ROB: robot.control_arm_joint_positions / _control_eef
        ENV-->>EV: get_obs() 经 preprocessor
    end
```

图依据：LejuRobotics/kuavo_data_challenge@c0ca6a4，读过 README.md、kuavo_deploy/readme/inference.md、kuavo_data/CvtRosbag2Lerobot.py、kuavo_train/train_policy.py（main 部分）、kuavo_deploy/src/scripts/script.py、kuavo_deploy/src/eval/real_single_test.py、kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [kuavo_data_challenge 源码归档](../../sources/repos/kuavo_data_challenge.md)（<https://github.com/LejuRobotics/kuavo_data_challenge>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LejuRobotics/kuavo_data_challenge@c0ca6a4，读过 README.md、kuavo_deploy/readme/inference.md、kuavo_data/CvtRosbag2Lerobot.py、kuavo_train/train_policy.py、kuavo_deploy/src/scripts/script.py、kuavo_deploy/src/eval/real_single_test.py、kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [乐聚机器人 官方组织](https://github.com/LejuRobotics)
