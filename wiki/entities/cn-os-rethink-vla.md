---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-10-12
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/rethink_vla.md
summary: "智在无界 开源项目 Rethink_VLA（VLA/操作模型）：VLA/策略接入：模型输出动作块驱动本体执行…"
institutions:
  - beingbeyond
---

# Rethink_VLA

## 一句话定义

**Rethink_VLA** 是 [智在无界](https://github.com/BeingBeyond) 公开的 **VLA/操作模型** 开源项目：VLA/策略接入：模型输出动作块驱动本体执行

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
| 机构 | 智在无界 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/BeingBeyond |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Rethink_VLA` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（评测）：`bash shell/eval-libero-relative.sh`，脚本先后台起 `python -m src.inference.run_server_vla`，再在 conda 环境 `libero` 中跑 `python -m src.benchmark.libero.run_libero_eval_fast`；两进程经 ZeroMQ（REQ/REP）通信。

```mermaid
sequenceDiagram
    participant SH as shell/eval-libero-relative.sh
    participant SRV as src/inference/run_server_vla.py
    participant POL as InternVLPolicy (internvl_policy.py)
    participant ZS as InternVLInferenceServer (ZMQ REP)
    participant EV as benchmark/libero/run_libero_eval_fast.py
    participant PC as Policy_Libero (benchmark/utils/policy.py)
    participant ENV as LIBERO env
    participant M as BeingH.get_action_bagel (model/beingvla.py)

    SH->>SRV: nohup python -m ... --model-path --port --data-config-name
    SRV->>POL: InternVLPolicy(...) 读 experiment_cfg 元数据与 *.safetensors
    SRV->>ZS: register get_action / get_modality_config 后 server.run()
    SH->>SH: 轮询日志直到出现 Server is ready
    SH->>EV: 对每个 libero_suite 启动评测进程
    EV->>PC: initialize_model -> Policy_Libero(port, exec_chunk_size)
    loop 每个 episode
        EV->>ENV: env.reset / set_init_state
        EV->>PC: policy.reset()
        loop t < max_steps
            EV->>PC: policy.get_action(obs_dict)
            alt 本地动作块用尽（t == exec_chunk_size）
                PC->>ZS: ExternalRobotInferenceClient.get_action (ZMQ REQ)
                ZS->>POL: policy.get_action(observations)
                POL->>POL: _modality_transform + _prepare_packed_inputs
                POL->>M: model.get_action_bagel(**packed_inputs)
                Note over M: 噪声动作 + Euler 积分，num_steps 次 forward_train 预测速度场
                M-->>POL: action_pred (chunk_length 步)
                POL-->>PC: 反归一化后的动作块
            end
            PC-->>EV: 第 t 步动作（eef_relative 变换为世界系）
            EV->>ENV: env.step(action)
        end
    end
```

图依据：BeingBeyond/Rethink_VLA@d3739ab，读过 `README.md`、`shell/eval-libero-relative.sh`、`shell/relative-post-libero-full-eef_relative-5shot.sh`、`src/inference/run_server_vla.py`、`src/inference/internvl_service.py`、`src/inference/service.py`、`src/inference/internvl_policy.py`、`src/benchmark/libero/run_libero_eval_fast.py`、`src/benchmark/utils/policy.py`、`src/model/beingvla.py`、`src/train/pretrain_unified_navit.py`。训练入口为 `torchrun src/train/pretrain_unified_navit.py`（`BeingH` + `PackedDataset` + FSDP，损失 `action_loss + und_loss`），未单独画图。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Rethink_VLA 源码归档](../../sources/repos/rethink_vla.md)（<https://github.com/BeingBeyond/Rethink_VLA>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：BeingBeyond/Rethink_VLA@d3739ab，读过 README.md、shell/eval-libero-relative.sh、src/inference/{run_server_vla,internvl_service,service,internvl_policy}.py、src/benchmark/libero/run_libero_eval_fast.py、src/benchmark/utils/policy.py、src/model/beingvla.py、src/train/pretrain_unified_navit.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [智在无界 官方组织](https://github.com/BeingBeyond)
