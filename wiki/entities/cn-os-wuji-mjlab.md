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
  - ../../sources/repos/wuji-mjlab.md
summary: "舞肌科技 开源项目 wuji-mjlab（移动操作）：基于mjlab训练Wuji Hand的手内物体旋转策略，公开PPO训练和Sim2Real部署入口，用于检查灵巧手接触控制从仿真到真机的完整链路。…"
institutions:
  - wuji-robotics
---

# wuji-mjlab

## 一句话定义

**wuji-mjlab** 是 [舞肌科技](https://github.com/wuji-technology) 公开的 **移动操作** 开源项目：基于mjlab训练Wuji Hand的手内物体旋转策略，公开PPO训练和Sim2Real部署入口，用于检查灵巧手接触控制从仿真到真机的完整链路。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第三层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 舞肌科技 |
| 类别 | 移动操作 |
| 官方组织 | https://github.com/wuji-technology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wuji-mjlab` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/train/train_rsl_rl.py --task WujiHand2_PenSpin`（`scripts/train/train_rsl_rl.py` 的 `main`；`pixi.toml` 的 `train` 任务即调用该脚本；安装见 README 的 pixi 步骤）

```mermaid
sequenceDiagram
    participant U as 用户 shell
    participant T as scripts/train/train_rsl_rl.py main
    participant R as wuji_mjlab/tasks 任务注册
    participant C as prepare_task_cfgs 与 TrainConfig
    participant L as _launch_training
    participant RUN as _run_train
    participant E as mjlab ManagerBasedRlEnv
    participant W as runner_cls（默认 WujiOnPolicyRunner）
    U->>T: python scripts/train/train_rsl_rl.py --task WujiHand2_PenSpin
    T->>R: import wuji_mjlab.tasks 经 import_packages 加载各任务，register_mjlab_task 登记
    T->>T: resolve_task 解析任务 ID，split_special_args 分离覆盖项
    T->>C: prepare_task_cfgs 生成 env 与 agent 配置
    T->>C: tyro.cli 以 TrainConfig.from_task 为默认值合并命令行参数
    T->>L: _launch_training(task_id, cfg)
    L->>L: select_gpus 选卡，设置 CUDA_VISIBLE_DEVICES 与 MUJOCO_GL=egl
    alt 单卡或 CPU
        L->>RUN: _run_train(task_id, cfg, log_dir)
    else 多卡
        L->>RUN: torchrunx Launcher 为每张卡启动 _run_train
    end
    RUN->>E: ManagerBasedRlEnv(cfg.env, device)
    RUN->>RUN: 可选 VideoRecorder，再经 RslRlVecEnvWrapper 包装环境
    RUN->>W: load_runner_cls 取得 runner 类并实例化
    RUN->>RUN: dump_yaml 写 params/env.yaml 与 agent.yaml，resume 时加载检查点
    RUN->>W: learn(num_learning_iterations)
    loop 每次 rollout
        W->>E: step(actions)，奖励与观测由任务配置的管理器计算
    end
```

图依据：wuji-mjlab@b2b98d7，README.md（任务表与安装步骤）、pixi.toml（仅 grep tasks 段）、scripts/train/train_rsl_rl.py（完整读过）、src/wuji_mjlab/tasks/__init__.py（读过）、src/wuji_mjlab/tasks/pen_spin/config/wuji_hand2/__init__.py（读过 register_mjlab_task 调用）、src/wuji_mjlab/tasks/pen_spin/config/wuji_hand2/env_cfgs.py（仅 grep 函数定义）。mjlab 库与 WujiOnPolicyRunner 内部实现不在本仓库且未读。deploy/ 与 play 流程未画。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wuji-mjlab 源码归档](../../sources/repos/wuji-mjlab.md)（<https://github.com/wuji-technology/wuji-mjlab>）

- 源码核对（2026-11-02）：wuji-technology/wuji-mjlab@b2b98d7，读过 README.md、scripts/train/train_rsl_rl.py、src/wuji_mjlab/tasks/__init__.py、src/wuji_mjlab/tasks/pen_spin/config/wuji_hand2/__init__.py；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [舞肌科技 官方组织](https://github.com/wuji-technology)
