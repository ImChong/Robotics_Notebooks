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
  - ../../sources/repos/robo-valuerl.md
summary: "北京人形机器人创新中心 开源项目 Robo-ValueRL（评测）：评测基准与工具…"
institutions:
  - x-humanoid
---

# Robo-ValueRL

## 一句话定义

**Robo-ValueRL** 是 [北京人形机器人创新中心](https://github.com/Open-X-Humanoid) 公开的 **评测** 开源项目：评测基准与工具

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
| 机构 | 北京人形机器人创新中心 |
| 类别 | 评测 |
| 官方组织 | https://github.com/Open-X-Humanoid |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Robo-ValueRL` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（均在 `robo_valuerl/` 下，README 的四阶段流程）：`bash scripts/train_robo_valuerl_value_estimator.sh` → `python generate_action_quality/annotate_value_function_sparse_part.py ...` → `python generate_action_quality/generate_quality.py --data_dir ...` → `bash scripts/train_robo_valuerl_offline.sh` → 部署 `python deploy_robo_valuerl.py` + `python rtc_env_client_inference_with_delta.py` → `bash scripts/train_robo_valuerl_online.sh`。

```mermaid
sequenceDiagram
    participant S1 as train_robo_valuerl_value_estimator.py
    participant C as OpenPi_X_Humanoid_Critic (agents/)
    participant A as annotate_value_function_sparse_part.py
    participant Q as generate_quality.py
    participant S3 as train_robo_value_rl_offline_rl.py
    participant SV as deploy_robo_valuerl.py (AgentServer)
    participant CL as rtc_env_client_inference_with_delta.py (RTCEnvClient)
    participant ENV as RealWorld_X_Humanoid_Env
    participant S4 as train_robo_value_rl_online_rl.py

    Note over S1,C: 阶段1 torchrun 8 卡，LeRobot v2.1 数据（history 配对）
    S1->>C: 构造 Critic，algorithm.step(batch) 训练 remain_time/value
    S1-->>A: 保存 step_N checkpoint（--pytorch-weight-path）
    Note over A,Q: 阶段2
    A->>C: 对每条 episode 每隔 stride 帧 predict_value(三相机+history_image+state)
    A->>A: 插值/平滑，写回 parquet 列 predict_remain_time 等
    Q->>Q: 读 predict_remain_time，按 50 帧块计算分数
    Q->>Q: 全局排名 → quality 0/1/2(BAD/MEDIUM/GOOD)，叠加人工干预覆盖与连续性降级
    Q->>Q: 覆写 parquet 的 quality 列，更新 meta/info.json
    Note over S3: 阶段3 配置 stack_all_hours_data_pretraining
    S3->>S3: MultiLeRobotDataset(delta_timestamps 含 quality) → XHumanoidLerobotOfflineWrapperForBC
    S3->>S3: loop: algorithm.step(batch)，PaligemmaTokenizerWithQuality 把 quality 放入提示
    Note over SV,ENV: 部署与 RTC 在线采集
    CL->>ENV: RealWorld_X_Humanoid_Env.prepare()，get_obs()
    CL->>SV: socket(8888) predict_rtc_action(obs, a_prev, d, s)
    alt a_prev 为空(首帧)
        SV->>SV: agent.predict_action(obs)
    else 有 a_prev
        SV->>SV: OpenPi_X_Humanoid_Agent_RTC.predict_rtc_action_with_noise(obs, a_prev, d)
    end
    SV-->>CL: 反归一化动作块(服务端把 action_quality 固定为 2)
    CL->>ENV: get_action() 逐步取动作 → env.step(action)
    Note over S4: 阶段4 在线 rollout 转 LeRobot 后微调
    S4->>S4: 离线数据 + rollout_repo_ids 数据拼接采样(rollout_ratio)，加载 --pytorch-weight-path
```

图依据：Open-X-Humanoid/Robo-ValueRL@1618bcb，读过 `README.md`、`install_simple.sh`、`robo_valuerl/scripts/train_robo_valuerl_offline.sh`、`scripts/train_robo_valuerl_online.sh`、`train_robo_valuerl_value_estimator.py`、`generate_action_quality/annotate_value_function_sparse_part.py`、`generate_action_quality/generate_quality.py`、`train_robo_value_rl_offline_rl.py`、`train_robo_value_rl_online_rl.py`、`deploy_robo_valuerl.py`、`rtc_env_client_inference_with_delta.py`。`agents/`、`algorithm/`、`openpi/` 内部实现与 `env_rtc_with_delta.py` 未读，Critic/Agent 的内部训练损失不在图中展开。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Robo-ValueRL 源码归档](../../sources/repos/robo-valuerl.md)（<https://github.com/Open-X-Humanoid/Robo-ValueRL>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Open-X-Humanoid/Robo-ValueRL@1618bcb，读过 README.md、install_simple.sh、robo_valuerl/ 下 scripts/*.sh、train_robo_valuerl_value_estimator.py、annotate_value_function_sparse_part.py、generate_quality.py、train_robo_value_rl_offline_rl.py、train_robo_value_rl_online_rl.py、deploy_robo_valuerl.py、rtc_env_client_inference_with_delta.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [北京人形机器人创新中心 官方组织](https://github.com/Open-X-Humanoid)
