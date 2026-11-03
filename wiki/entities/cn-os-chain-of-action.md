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
  - ../../sources/repos/chain-of-action.md
summary: "字节跳动机器人团队 开源项目 Chain-of-Action（VLA/操作模型）：面向机器人操作的轨迹自回归建模（动作链推理）…"
institutions:
  - bytedance
---

# Chain-of-Action

## 一句话定义

**Chain-of-Action** 是 [字节跳动机器人团队](https://github.com/ByteDance-Seed) 公开的 **VLA/操作模型** 开源项目：面向机器人操作的轨迹自回归建模（动作链推理）

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
| 机构 | 字节跳动机器人团队 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/ByteDance-Seed、https://github.com/bytedance |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Chain-of-Action` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：训练 `python scripts/train.py task=<task>`；评测 `bash scripts/eval.sh task=<task>`（内部先 `scripts/download_pretrained_model.py`、`scripts/download_dataset.py`，再 `xvfb-run -a python scripts/eval.py snapshot=...`）。

```mermaid
sequenceDiagram
    participant U as scripts/train.py · eval.py (hydra main)
    participant W as src/workspace.py<br/>Workspace
    participant D as RLBenchDataset<br/>(src/dataset)
    participant A as CoA (src/methods/coa/coa.py)
    participant AM as ActorModel + transformer.py
    participant E as RLBench 评测环境<br/>(RLBenchEnvFactory.make_eval_env)

    U->>W: Workspace(cfg, train)；eval.py 先 torch.load(snapshot) 并并入 ckpt 配置
    W->>D: env_factory._load_demos → RLBenchDataset(seq_len=action_sequence)（仅 train）
    W->>A: hydra.utils.instantiate(cfg.method)；有 snapshot 则 load_snapshot
    W->>E: make_eval_env(cfg)
    alt 训练：train.py → workspace.loop()
        loop num_train_steps
            W->>D: 下一批 batch (RandomSampler)
            W->>A: agent.update(batch)
            A->>AM: forward(training=True)：encoder_model 取图像特征 → actor_model(obs_feat, proprio, a_gt, is_pad)
            AM-->>A: a_hat, x_hat, x_gt
            A->>A: _compute_loss (action_loss + latent_loss) → backward / opt.step
            W->>W: 每 eval_every_steps 调 eval()；每 save_every_steps 调 save_snapshot()
        end
    else 评测：eval.py → workspace.eval()
        loop 每个 episode (low_dim_obs.pkl)
            W->>E: eval_env.reset_to_demo(pkl)
            loop 直到 termination / truncation
                W->>A: agent.act(obs)（_perform_env_steps）
                A->>AM: forward(training=False) → a_hat；按 action_order 翻转/滚动
                A-->>W: 动作序列 (Batch, T, action_dim)
                W->>E: env.step(action)
            end
        end
        W->>W: 汇总 success_rate 等指标并记录
    end
```

图依据：ByteDance-Seed/Chain-of-Action@49d0710，读过 README.md、scripts/eval.sh、scripts/eval.py、scripts/train.py、src/cfgs/launch.yaml（前 50 行）、src/workspace.py、src/methods/coa/coa.py（forward/act/_compute_loss/update 与 ActorModel.forward）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Chain-of-Action 源码归档](../../sources/repos/chain-of-action.md)（<https://github.com/ByteDance-Seed/Chain-of-Action>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：ByteDance-Seed/Chain-of-Action@49d0710，读过 README.md、scripts/eval.sh、scripts/eval.py、scripts/train.py、src/cfgs/launch.yaml（前 50 行）、src/workspace.py、src/methods/coa/coa.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [字节跳动机器人团队 官方组织](https://github.com/ByteDance-Seed)
