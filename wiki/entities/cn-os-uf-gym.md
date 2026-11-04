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
  - ../../sources/repos/uf-gym.md
summary: "众为创造 开源项目 uf-gym（运动RL/技能训练）：在panda-gym基础上加入UFACTORY机器人模型和到达、抓取放置等任务，用于机械臂强化学习的快速仿真实验。…"
institutions:
  - zhongwei
---

# uf-gym

## 一句话定义

**uf-gym** 是 [众为创造](https://github.com/xArm-Developer) 公开的 **运动RL/技能训练** 开源项目：在panda-gym基础上加入UFACTORY机器人模型和到达、抓取放置等任务，用于机械臂强化学习的快速仿真实验。

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
| 机构 | 众为创造 |
| 类别 | 运动RL/技能训练 |
| 官方组织 | https://github.com/xArm-Developer |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `uf-gym` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python train_xarm6_reach.py`（需要与本仓库同级的 `../panda-gym`、stable-baselines3 与 pybullet；测试入口为 `python test_xarm6_reach.py`）

```mermaid
sequenceDiagram
    autonumber
    participant U as 操作者（shell）
    participant TS as train_xarm6_reach.py
    participant GR as uf_gym/__init__.py（gym.register）
    participant UE as XArm6ReachEnv（uf_gym/envs/uf_tasks/xarm6_tasks.py）
    participant XR as XArm6（uf_gym/envs/uf_robots/xarm6.py）
    participant PG as panda-gym（RobotTaskEnv、PyBullet、Reach，仓库外）
    participant SB as stable-baselines3 DDPG 与 HerReplayBuffer（仓库外）
    participant FS as model/ 与 logs/

    U->>TS: python train_xarm6_reach.py
    TS->>PG: sys.path 加入 ../panda-gym，import panda_gym
    TS->>GR: import uf_gym，循环注册 XArm6Reach-v3 等环境 ID
    TS->>GR: gym.make("XArm6Reach-v3", render_mode="rgb_array")
    GR->>UE: 按 entry_point uf_gym.envs:XArm6ReachEnv 构造，reward sparse，control ee
    UE->>PG: PyBullet(render_mode, renderer)
    UE->>XR: XArm6(sim, block_gripper=True, base_position=[-0.5, 0, 0])
    UE->>PG: Reach(sim, reward_type, get_ee_position)
    UE->>PG: super().__init__(robot, task, 渲染参数)，组成 RobotTaskEnv
    TS->>SB: DDPG(超参, env, HerReplayBuffer, tensorboard_log)
    TS->>SB: model.learn(500_000)
    loop 每个环境步
        SB->>PG: env.step(action)
        PG->>XR: set_action，裁剪后经 ee_displacement_to_target_arm_angles 转为关节目标
        PG->>PG: 推进 PyBullet 物理仿真
        PG->>XR: get_obs，取末端位置、速度与可选手指宽度
        PG->>PG: 任务计算目标与奖励
        PG-->>SB: obs、reward、done、info
    end
    TS->>FS: model.save('./model/ddpg-XArm6Reach-v3.pkl')
```

图依据：uf-gym@5d93e04，train_xarm6_reach.py（全文）、test_xarm6_reach.py（全文）、uf_gym/__init__.py（全文）、uf_gym/envs/__init__.py（全文）、uf_gym/envs/uf_robots/xarm6.py（set_action、get_obs、reset 与构造段）、uf_gym/envs/uf_tasks/xarm6_tasks.py（XArm6ReachEnv 段）、README.md

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [uf-gym 源码归档](../../sources/repos/uf-gym.md)（<https://github.com/xArm-Developer/uf-gym>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：xArm-Developer/uf-gym@5d93e04，读过 README.md、train_xarm6_reach.py、test_xarm6_reach.py、uf_gym/__init__.py、uf_gym/envs/__init__.py、uf_gym/envs/uf_robots/xarm6.py（set_action、get_obs、reset 与构造段）、uf_gym/envs/uf_tasks/xarm6_tasks.py（XArm6ReachEnv 段）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [众为创造 官方组织](https://github.com/xArm-Developer)
