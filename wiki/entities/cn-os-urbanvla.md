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
  - ../../sources/repos/urbanvla.md
summary: "银河通用 开源项目 UrbanVLA（VLA/操作模型）：将第一视角视觉、语言指令与机器人状态映射为移动决策，使机器人在室外或半开放城市环境中完成目标导向导航。…"
institutions:
  - galbot
---

# UrbanVLA

## 一句话定义

**UrbanVLA** 是 [银河通用](https://github.com/GalaxyGeneralRobotics) 公开的 **VLA/操作模型** 开源项目：将第一视角视觉、语言指令与机器人状态映射为移动决策，使机器人在室外或半开放城市环境中完成目标导向导航。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第二层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 银河通用 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/GalaxyGeneralRobotics、https://github.com/PKU-EPIC |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `UrbanVLA` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

官方仓库当前公开的是数据采集、后处理与 MetaUrban 基准评测流水线（README 称模型结构、训练代码与 checkpoint 尚未发布），下图画的是评测主循环。

入口：`MAX_EPISODES=1 NO_VISUALIZATION=1 bash benchmark/eval.sh pointnav-test 0`（默认 `POLICY=dummy`；`POLICY=metaurban_ppo` 可跑自带 PPO 专家）

```mermaid
sequenceDiagram
    participant SH as benchmark/eval.sh
    participant SMP as benchmark/eval_sampling.py
    participant EV as benchmark/metaurban_eval.py
    participant POL as benchmark/policy (build_policy → NavigationPolicy)
    participant ENV as metaurban.envs Sidewalk*MetaUrbanEnv
    participant NOISE as data_postprocess/common (PolylineProcessor、normalization)
    SH->>SMP: python -m benchmark.eval_sampling 得到 X/C/S 各自 episode 数
    SH->>EV: 按 map 与 worker id 启动 python benchmark/metaurban_eval.py（每 GPU 一个进程）
    EV->>EV: 读 config/config_static.yaml 或 config_dynamic.yaml，select_gpu
    EV->>POL: build_policy(--policy)，按名字懒加载 benchmark/policy/<name>.py
    EV->>ENV: env_type(config)，写 run_manifest.json
    loop 每个 episode
        EV->>ENV: env.reset(seed=12800+episode_id)
        ENV-->>EV: obs（ego_state、参考轨迹 traj、next_ten_ckpts）
        EV->>NOISE: PolylineProcessor.simulate_sd_noise_pipeline(traj) 得到转角 corners/turns
        loop 直到 done 或 truncated
            EV->>EV: 由最近检查点推出 next_action（stop/turn left/turn right）与 duration
            EV->>POL: predict(observations, {traj_to_go, next_action, duration, ego_state})
            POL-->>EV: 8 个归一化 [dx, dy, dtheta]
            EV->>NOISE: denormalize_supervision(NORM_SCALE_X/Y/THETA)
            EV->>EV: 相邻点插值成 15 点，局部系转世界系
            EV->>ENV: set_position / set_heading_theta，再 env.step([0.0, 0.0])
            ENV-->>EV: obs, info（crash_*、is_success、route_completion）
        end
        EV->>EV: 写 info.json 与 metric.json（sns、sr、spl、rc、cost）
    end
```

图依据：GalaxyGeneralRobotics/UrbanVLA@c75294a，读过 README.md、evaluation.md、benchmark/eval.sh、benchmark/metaurban_eval.py、benchmark/policy/__init__.py、benchmark/policy/dummy.py、scripts/collect.py、scripts/rl_expert_collect.py。注：`control_mode="env_action"` 的策略（如 metaurban_ppo）走 `predict_action` 并直接 `env.step(action)`，图中未画此分支。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [UrbanVLA 源码归档](../../sources/repos/urbanvla.md)（<https://github.com/GalaxyGeneralRobotics/UrbanVLA>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：GalaxyGeneralRobotics/UrbanVLA@c75294a，读过 README.md、evaluation.md、benchmark/eval.sh、benchmark/metaurban_eval.py、benchmark/policy/__init__.py、benchmark/policy/dummy.py、scripts/collect.py、scripts/rl_expert_collect.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [银河通用 官方组织](https://github.com/GalaxyGeneralRobotics)
