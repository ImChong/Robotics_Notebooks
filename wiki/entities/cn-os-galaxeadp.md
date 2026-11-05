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
  - ../../sources/repos/galaxeadp.md
summary: "星海图 开源项目 GalaxeaDP（VLA/操作模型）：把相机观测、机器人状态和任务条件映射为连续动作块，用扩散策略完成双臂或移动操作；项目适合作为GalaxeaVLA之外的模仿学习基线。…"
institutions:
  - galaxea
---

# GalaxeaDP

## 一句话定义

**GalaxeaDP** 是 [星海图](https://github.com/OpenGalaxea) 公开的 **VLA/操作模型** 开源项目：把相机观测、机器人状态和任务条件映射为连续动作块，用扩散策略完成双臂或移动操作；项目适合作为GalaxeaVLA之外的模仿学习基线。

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
| 机构 | 星海图 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/OpenGalaxea |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `GalaxeaDP` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：一键流程 `bash scripts/sim_r1pro_blocks.sh`，依次执行 GalaxeaManipSim 的数据采集/回放/转换脚本，然后 `bash scripts/train.sh task=sim/<task> ...`（内部 `python src/train.py`）、`bash scripts/eval_open_loop.sh`（`src/eval_open_loop.py`）、`bash scripts/eval_sim.sh`（`src/eval_sim.py`）。

```mermaid
sequenceDiagram
    participant SH as scripts/sim_r1pro_blocks.sh
    participant SIM as galaxea_sim.scripts (GalaxeaManipSim)
    participant TR as src/train.py (Hydra + Lightning)
    participant DM as src/data/galaxea_lerobot_dataset.py
    participant MOD as src/models/dp_bc_module.py DiffusionPolicyBCModule
    participant POL as src/models/policy/diffusion_unet_image_policy.py
    participant EV as src/eval_sim.py
    participant ENV as gym.make(env)
    SH->>SIM: collect_demos 再 replay_demos 再 convert_single_galaxea_sim_to_galaxea_lerobot
    SIM-->>SH: dataset_dir/env/lerobot
    SH->>TR: bash scripts/train.sh task=sim/task
    TR->>DM: hydra.utils.instantiate(cfg.data)
    TR->>MOD: instantiate(cfg.model)，setup() 里 policy.set_normalizer(get_normalizer())
    TR->>TR: trainer.fit(model, datamodule)
    loop 训练步
        MOD->>POL: compute_loss(batch)：obs_encoder 编码观测，加噪，预测噪声或样本
    end
    SH->>EV: bash scripts/eval_sim.sh task=... ckpt_path=... env=... target_controller_type=...
    EV->>MOD: instantiate(cfg.model)，load_state_dict(ckpt)，取 model.policy.eval()
    EV->>ENV: gym.make(env, control_freq=30, controller_type)
    loop 每个评测 episode
        EV->>ENV: reset(seed=42)
        loop 直到 terminated 或 truncated
            alt action_seq 用完
                EV->>EV: make_single_frame_batch_eef 或 make_single_frame_batch_joints（图像缩放 224x224）
                EV->>POL: predict_action(batch)，conditional_sample 做 DDPM 去噪
                POL-->>EV: 动作序列，取前 num_action_steps(16) 步
            end
            EV->>ENV: env.step(action_seq[seq_idx])
        end
    end
```

图依据：OpenGalaxea/GalaxeaDP@21abf51，读过 `README.md`、`scripts/sim_r1pro_blocks.sh`、`scripts/train.sh`、`scripts/eval_sim.sh`、`src/train.py`、`src/eval_sim.py`、`src/models/dp_bc_module.py`、`src/models/policy/diffusion_unet_image_policy.py`（类/方法骨架）、`src/data/galaxea_lerobot_dataset.py`（方法骨架）；`src/eval_open_loop.py`、obs_encoder 与 UNet 内部未读，`compute_loss` 内部细节按方法名与调用行推断。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [GalaxeaDP 源码归档](../../sources/repos/galaxeadp.md)（<https://github.com/OpenGalaxea/GalaxeaDP>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：OpenGalaxea/GalaxeaDP@21abf51，读过 README.md、scripts/sim_r1pro_blocks.sh、scripts/train.sh、scripts/eval_sim.sh、src/train.py、src/eval_sim.py、src/models/dp_bc_module.py、src/models/policy/diffusion_unet_image_policy.py、src/data/galaxea_lerobot_dataset.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [星海图 官方组织](https://github.com/OpenGalaxea)
