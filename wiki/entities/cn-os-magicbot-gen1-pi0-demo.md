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
  - ../../sources/repos/magicbot-gen1_pi0_demo.md
summary: "魔法原子 开源项目 magicbot-gen1_pi0_demo（VLA/操作模型）：展示pi0类策略接入MagicBot Gen1的模型、观测与执行接口，为VLA到真机动作链路提供示例。…"
institutions:
  - magiclab
---

# magicbot-gen1_pi0_demo

## 一句话定义

**magicbot-gen1_pi0_demo** 是 [魔法原子](https://github.com/MagiclabRobotics) 公开的 **VLA/操作模型** 开源项目：展示pi0类策略接入MagicBot Gen1的模型、观测与执行接口，为VLA到真机动作链路提供示例。

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
| 机构 | 魔法原子 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/MagiclabRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `magicbot-gen1_pi0_demo` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（三步，均在 README 给出）：`uv run scripts/compute_norm_stats.py` → `uv run scripts/train.py pi0_magiclab_low_mem_finetune --exp-name=...` → `uv run scripts/serve_policy.py policy:checkpoint --policy.config=... --policy.dir=...`。推理侧机器人客户端（`pico_teleoperate` 采集、真机部署）不在本仓库内；本图只画仓库内可见的训练与策略服务调用链，数据转换脚本 `examples/libero/convert_magiclab_humanoid_data_to_lerobot.py` 仅按 README 描述，未读其内部。

```mermaid
sequenceDiagram
    participant N as scripts/compute_norm_stats.py
    participant T as scripts/train.py main()
    participant D as openpi.training.data_loader
    participant C as openpi.training.config (TrainConfig)
    participant CK as openpi.training.checkpoints
    participant S as scripts/serve_policy.py
    participant PC as openpi.policies.policy_config
    participant W as WebsocketPolicyServer
    participant Pol as openpi.policies.policy.Policy
    participant Cli as 外部客户端（openpi_client，不在本图内部）
    N->>C: get_config("pi0_magiclab_low_mem_finetune")
    N->>D: create_dataset + TransformedDataset（repack/data transforms，去掉字符串）
    N->>N: 遍历 state / actions，RunningStats.update
    N->>N: normalize.save(OUTPUT_PATH/repo_id, norm_stats)
    T->>CK: initialize_checkpoint_dir(checkpoint_dir, resume, overwrite)
    T->>D: create_data_loader(config, sharding, shuffle=True)
    T->>T: init_train_state（_load_weights_and_validate 加载权重）
    loop step in range(num_train_steps)
        T->>T: ptrain_step(train_rng, train_state, batch)（jax.jit 的 train_step）
        T->>D: next(data_iter)
        T->>CK: save_state（每 save_interval 步及最后一步）
    end
    S->>C: get_config(policy.config)
    S->>PC: create_trained_policy(config, policy.dir)
    PC->>PC: restore_params(checkpoint_dir/params) 并 model.load
    PC->>CK: load_norm_stats(checkpoint_dir/assets, asset_id)
    PC-->>S: Policy
    S->>W: WebsocketPolicyServer(policy, host=0.0.0.0, port=1077).serve_forever()
    Cli->>W: websocket 发送 msgpack 观测 obs
    W->>Pol: infer(obs)
    Pol->>Pol: input_transform → model.sample_actions → output_transform
    W-->>Cli: msgpack 动作 action
```

图依据：magicbot-gen1_pi0_demo@3e6feb5，读过 README.md、scripts/compute_norm_stats.py、scripts/train.py（main）、scripts/serve_policy.py、src/openpi/policies/policy_config.py、src/openpi/policies/policy.py（infer）、src/openpi/serving/websocket_policy_server.py。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [magicbot-gen1_pi0_demo 源码归档](../../sources/repos/magicbot-gen1_pi0_demo.md)（<https://github.com/MagiclabRobotics/magicbot-gen1_pi0_demo>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：MagiclabRobotics/magicbot-gen1_pi0_demo@3e6feb5，读过 README.md、scripts/compute_norm_stats.py、scripts/train.py、scripts/serve_policy.py、src/openpi/policies/policy_config.py、src/openpi/policies/policy.py、src/openpi/serving/websocket_policy_server.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [魔法原子 官方组织](https://github.com/MagiclabRobotics)
