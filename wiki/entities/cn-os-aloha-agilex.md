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
  - ../../sources/repos/aloha-agilex.md
summary: "松灵机器人 开源项目 aloha-agilex（具身Agent/规划）：移动操作（Loco-Manip）策略接入…"
institutions:
  - agilex
---

# aloha-agilex

## 一句话定义

**aloha-agilex** 是 [松灵机器人](https://github.com/agilexrobotics) 公开的 **具身Agent/规划** 开源项目：移动操作（Loco-Manip）策略接入

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
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/agilexrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `aloha-agilex` 仓库并核对 README 许可与依赖。
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

入口（训练）：`cd act && python aloha_train.py --dataset_dir <hdf5 数据目录> --ckpt_dir <检查点目录>`；入口（推理，需先启动 piper 机械臂 ROS 节点）：`python aloha_inference-ros2.py --ckpt_dir <检查点目录>`（均出自仓库根 `readme.md`）

```mermaid
sequenceDiagram
    participant TR as act/aloha_train.py
    participant U as act/utils.py load_data
    participant P as act/policy.py ACTPolicy
    participant CK as ckpt_dir（policy_best.ckpt、dataset_stats.pkl）
    participant INF as act/aloha_inference-ros2.py RosOperator(Node)
    participant TH as inference_process 线程
    participant ROS as ROS2 话题（相机、关节、末端位姿）与 piper 节点

    TR->>U: load_data(args) 读取 hdf5 数据集，返回 train/val dataloader 与 stats
    TR->>CK: pickle.dump(stats) 写 dataset_stats.pkl
    loop num_epochs（train_process）
        TR->>P: forward_pass(policy, data)，val 阶段先 eval
        P-->>TR: loss_dict
        TR->>TR: loss.backward、optimizer.step、lr_scheduler.step
    end
    TR->>CK: torch.save(best_state_dict) 写 policy_best.ckpt
    Note over INF: 推理为另一个进程
    INF->>INF: main：rclpy.init，RosOperator(args)，rclpy.spin
    INF->>INF: __init__ 创建各 subscription / publisher，并起 model_inference 线程
    INF->>CK: torch.load(ckpt) 与 pickle.load(stats)，policy.deserialize，policy.cuda().eval()
    ROS-->>INF: 彩色/深度/点云图像、arm_joint_state、arm_end_pose 回调缓存
    loop while t < max_publish_step
        INF->>TH: threading.Thread(inference_process)
        TH->>TH: pre_process 观测，取最近 history 帧
        TH->>P: policy(camera_color, ..., qpos...) 得到 all_actions
        TH->>TH: post_process（反归一化）写入全局 inference_actions
        INF->>INF: 取动作（可选 temporal_agg）
        INF-->>ROS: arm_joint_state_ctrl / arm_end_pose_ctrl 线性插值后发布
    end
```

图依据：agilexrobotics/aloha-agilex@275cbe9，读过 `readme.md`、`act/README.md`、`act/aloha_train.py`（`train` / `train_process` / `forward_pass`）、`act/aloha_inference-ros2.py`（`main` / `RosOperator.__init__` / `model_inference` / `inference_process` 片段）、`act/utils.py`（仅 `find_all_hdf5` 附近）。`act/policy.py`、`detr/`、`cleandiffuser/` 等模型内部未读；推理循环中动作取用与发布的细节为按片段概括。

## 参考来源

- [aloha-agilex 源码归档](../../sources/repos/aloha-agilex.md)（<https://github.com/agilexrobotics/aloha-agilex>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：agilexrobotics/aloha-agilex@275cbe9，读过 readme.md、act/README.md、act/aloha_train.py、act/aloha_inference-ros2.py、act/utils.py（部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [松灵机器人 官方组织](https://github.com/agilexrobotics)
