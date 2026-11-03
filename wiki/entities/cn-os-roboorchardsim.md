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
  - ../../sources/repos/roboorchardsim.md
summary: "地平线 开源项目 RoboOrchardSim（仿真环境）：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证…"
institutions:
  - horizon-robotics
---

# RoboOrchardSim

## 一句话定义

**RoboOrchardSim** 是 [地平线](https://github.com/HorizonRobotics) 公开的 **仿真环境** 开源项目：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第四层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 地平线 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/HorizonRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `RoboOrchardSim` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python3 examples/manipulation-app/scripts/eval_policy.py --eval-config examples/manipulation-app/configs/eval_example.yaml --output-dir eval_result/run_001 --gpus 0`（主进程为调度器，通过子进程再次调用同一脚本跑每个分片）

```mermaid
sequenceDiagram
    participant U as 用户 CLI
    participant EP as eval_policy.py main/_dispatch
    participant DW as _dispatch_work_items
    participant SW as _run_shard_worker (线程池)
    participant RS as execution.run_shard (子进程)
    participant EV as evaluator.Evaluator
    participant L as launcher.SimpleIsaacAppLauncher
    participant PF as policy.factory
    participant BM as benchmark.build_task

    U->>EP: --eval-config / --output-dir / --gpus
    EP->>EP: load_eval_config（展开 ${ORCHARD_ASSET}）
    EP->>DW: _dispatch(eval_cfg, args)
    DW->>DW: _build_plan 生成 BatchPlan
    DW->>DW: allocate_task_shards / build_evaluation_shards
    DW->>DW: write_evaluation_manifest（evaluation_manifest.json）
    loop 每个分片（按估算耗时降序）
        DW->>SW: 线程池提交
        SW->>RS: subprocess.Popen(python eval_policy.py --_shard-manifest ... --_shard-id ...)，设 CUDA_VISIBLE_DEVICES
        RS->>EP: main() 读 manifest，_load_model_cfg
        RS->>L: evaluation_runtime() 启动 Isaac 应用
        RS->>PF: create_policy_from_model_cfg
        PF-->>RS: dummy/groot/holobrain/motus/server 等策略
        RS->>EV: evaluator_cfg() 然后 run_with_runtime(policy, runtime)
        loop 每个场景种子
            EV->>BM: build_task(task_name, resolver, config_path)
            BM-->>EV: OrchardEnv，to_isaac_env_cfg 后开 IsaacEnvContextManager
            loop 每个 episode
                EV->>EV: _prepare_episode（env.reset、validator.reset、policy.reset）
                loop 至 max_steps 或 success/terminated/truncated
                    EV->>EV: canonicalize_observations 生成策略输入
                    EV->>RS: action = policy(policy_input)
                    EV->>EV: env.step(action)，validator.evaluate
                end
            end
        end
        RS->>RS: 写 eval_result.json / shard_summary.json
    end
    DW->>DW: _merge_task_shard_summaries，_write_summary（summary.json）
```

图依据：HorizonRobotics/RoboOrchardSim@f2e041e，读过 README.md、pyproject.toml、examples/manipulation-app/scripts/eval_policy.py、examples/manipulation-app/configs/eval_example.yaml、robo_orchard_sim/pipeline/evaluator/execution.py、robo_orchard_sim/pipeline/evaluator/evaluator.py、robo_orchard_sim/policy/factory.py。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RoboOrchardSim 源码归档](../../sources/repos/roboorchardsim.md)（<https://github.com/HorizonRobotics/RoboOrchardSim>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：HorizonRobotics/RoboOrchardSim@f2e041e，读过 README.md、pyproject.toml、examples/manipulation-app/scripts/eval_policy.py、configs/eval_example.yaml、robo_orchard_sim/pipeline/evaluator/{execution,evaluator}.py、robo_orchard_sim/policy/factory.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [地平线 官方组织](https://github.com/HorizonRobotics)
