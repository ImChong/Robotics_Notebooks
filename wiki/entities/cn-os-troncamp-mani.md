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
  - ../../sources/repos/troncamp-mani.md
summary: "逐际动力 开源项目 troncamp-mani（具身Agent/规划）：移动操作（Loco-Manip）策略接入…"
institutions:
  - limx
---

# troncamp-mani

## 一句话定义

**troncamp-mani** 是 [逐际动力](https://github.com/limxdynamics) 公开的 **具身Agent/规划** 开源项目：移动操作（Loco-Manip）策略接入

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
| 机构 | 逐际动力 |
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/limxdynamics、https://github.com/FluxVLA |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `troncamp-mani` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`bash collect_data.sh adjust_bottle adjust_bottle_200ep 0`（自采数据路径；训练与评测路径未在本图内）

```mermaid
sequenceDiagram
    autonumber
    participant U as 选手（shell）
    participant CS as collect_data.sh（仓库根）
    participant CT as recipes/rollout/collect_tron2_data.py
    participant PT as tron2_runtime_patch.py
    participant RC as RoboTwin script/collect_data.py（main 与 run）
    participant EX as envs/任务.py（专家 play_once、check_success）
    participant BT as envs/_base_task.py（Base_Task）
    participant SP as SAPIEN 与 cuRobo 规划（仓库外依赖）
    participant FS as save_path（轨迹数据目录）

    U->>CS: bash collect_data.sh task_name task_config gpu_id
    CS->>CS: 设置 CUDA_VISIBLE_DEVICES 与 ROBOTWIN_SAVE_DURING_SEARCH=1
    CS->>CT: python recipes/rollout/collect_tron2_data.py task_name task_config
    CT->>CT: 将 external/robotwin_local 加入 sys.path 并 chdir，spawn 启动方式
    CT->>PT: apply_tron2_runtime_patches()
    Note over PT: 关闭机器人自碰撞、重调手臂驱动、校验末端抓取可达性等补丁
    CT->>RC: collect_main(task_name, task_config)
    RC->>RC: 读取 task_config/<config>.yml 与 embodiment 配置，拼接 save_path
    RC->>EX: class_decorator 实例化任务类
    RC->>EX: run() 先以 seed=0 调用 setup_demo 预热 cuRobo 规划器
    loop 搜索直到成功数达到 episode_num
        RC->>EX: setup_demo(now_ep_num, seed=epid)
        RC->>EX: play_once() 专家生成并执行轨迹
        EX->>SP: 规划与 take_action(qpos)
        RC->>EX: plan_success 且 check_success()
        alt 成功
            RC->>BT: save_traj_data(suc_num)
            BT->>FS: 写入轨迹数据，记录 seed
        else 失败
            RC->>RC: 记录失败，换下一个 seed
        end
    end
    RC->>FS: 写 seed.txt 与 scene_info.json
    RC->>RC: os.system 调用 gen_episode_instructions.sh 生成指令
```

图依据：limxdynamics/troncamp-mani@7630a1a，collect_data.sh、recipes/rollout/collect_tron2_data.py（全文）、recipes/rollout/tron2_runtime_patch.py（函数列表与 apply_tron2_runtime_patches 段）、external/robotwin_local/script/collect_data.py（main 与 run 关键段）、external/robotwin_local/envs/adjust_bottle.py 与 envs/_base_task.py（函数列表）、README.md

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [troncamp-mani 源码归档](../../sources/repos/troncamp-mani.md)（<https://github.com/limxdynamics/troncamp-mani>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：limxdynamics/troncamp-mani@7630a1a，读过 README.md、collect_data.sh、recipes/rollout/collect_tron2_data.py、recipes/rollout/tron2_runtime_patch.py（函数列表与 apply_tron2_runtime_patches 段）、external/robotwin_local/script/collect_data.py（main 与 run 关键段）、external/robotwin_local/envs/adjust_bottle.py 与 external/robotwin_local/envs/_base_task.py（函数列表）；图中仅画自采数据路径，训练与评测路径未画；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [逐际动力 官方组织](https://github.com/limxdynamics)
